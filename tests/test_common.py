"""
Unit tests for common module (data loader, config, evaluation).
"""

import pytest
from common.config import LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES
from common.data_loader import (
    validate_statement,
    validate_label,
    label_to_idx,
    build_metadata_text,
    load_liar,
    get_test_indices,
)
from common.evaluate import evaluate, build_results_dict


class TestConfig:
    """Test configuration constants."""

    def test_label_mapping(self):
        """Test label to index mapping."""
        assert LABEL_TO_IDX["pants-fire"] == 0
        assert LABEL_TO_IDX["false"] == 1
        assert LABEL_TO_IDX["barely-true"] == 2
        assert LABEL_TO_IDX["half-true"] == 3
        assert LABEL_TO_IDX["mostly-true"] == 4
        assert LABEL_TO_IDX["true"] == 5

    def test_num_classes(self):
        """Test number of classes."""
        assert NUM_CLASSES == 6

    def test_bidirectional_mapping(self):
        """Test that IDX_TO_LABEL is inverse of LABEL_TO_IDX."""
        for label, idx in LABEL_TO_IDX.items():
            assert IDX_TO_LABEL[idx] == label


class TestValidation:
    """Test validation functions."""

    def test_validate_statement_valid(self):
        """Test valid statements."""
        assert validate_statement("This is a valid statement.")
        assert validate_statement("A")
        assert validate_statement("x" * 500)

    def test_validate_statement_invalid(self):
        """Test invalid statements."""
        assert not validate_statement("")
        assert not validate_statement("   ")
        assert not validate_statement("x" * 1001)
        assert not validate_statement(None)
        assert not validate_statement(123)

    def test_validate_label_valid(self):
        """Test valid labels."""
        assert validate_label("pants-fire")
        assert validate_label("PANTS-FIRE")
        assert validate_label("False")
        assert validate_label("true")

    def test_validate_label_invalid(self):
        """Test invalid labels."""
        assert not validate_label("invalid")
        assert not validate_label("maybe")
        assert not validate_label("")


class TestLabelConversion:
    """Test label to index conversion."""

    def test_label_to_idx_valid(self):
        """Test valid label conversions."""
        assert label_to_idx("pants-fire") == 0
        assert label_to_idx("false") == 1
        assert label_to_idx("true") == 5

    def test_label_to_idx_case_insensitive(self):
        """Test that conversion is case-insensitive."""
        assert label_to_idx("TRUE") == label_to_idx("true")
        assert label_to_idx("PANTS-FIRE") == label_to_idx("pants-fire")

    def test_label_to_idx_hf_integer_ids(self):
        """HF integer ids map by label name, not by raw value."""
        assert label_to_idx(0) == LABEL_TO_IDX["false"]
        assert label_to_idx(3) == LABEL_TO_IDX["true"]
        assert label_to_idx(4) == LABEL_TO_IDX["barely-true"]
        assert label_to_idx(5) == LABEL_TO_IDX["pants-fire"]

    def test_label_to_idx_invalid_integer(self):
        """Out-of-range ids are rejected."""
        with pytest.raises(ValueError):
            label_to_idx(6)
        with pytest.raises(ValueError):
            label_to_idx(-1)
        assert not validate_label(6)

    def test_label_to_idx_invalid(self):
        """Test that invalid labels raise ValueError."""
        with pytest.raises(ValueError):
            label_to_idx("invalid_label")
        with pytest.raises(ValueError):
            label_to_idx("maybe")


class TestMetadata:
    """Test metadata building."""

    def test_build_metadata_text(self):
        """Test metadata text building."""
        row = {
            "speaker": "John Doe",
            "party_affiliation": "Democrat",
            "subject": "healthcare",
            "credit_history": [1, 2, 3, 4, 5],
            "statement": "Healthcare is important."
        }
        result = build_metadata_text(row)

        assert "speaker: John Doe" in result
        assert "party: Democrat" in result
        assert "subject: healthcare" in result
        assert "1,2,3,4,5" in result
        assert "Healthcare is important." in result
        assert "[SEP]" in result

    def test_build_metadata_text_missing_fields(self):
        """Test that missing fields are handled gracefully."""
        row = {
            "statement": "Test statement."
        }
        result = build_metadata_text(row)

        assert "UNKNOWN" in result
        assert "Test statement." in result


class TestMetadataRealSchema:
    """Regression tests: the real HF LIAR rows use 5 separate *_counts columns."""

    REAL_ROW = {
        "statement": "Says the Annies List group supports third-trimester abortions.",
        "subject": "abortion",
        "speaker": "dwayne-bohac",
        "party_affiliation": "republican",
        "barely_true_counts": 3.0,
        "false_counts": 1.0,
        "half_true_counts": 4.0,
        "mostly_true_counts": 2.0,
        "pants_on_fire_counts": 7.0,
    }

    def test_history_uses_real_count_columns(self):
        """History must come from the 5 *_counts columns (order: barely, false, half, mostly,
        pants-on-fire), not fall back to zeros."""
        text = build_metadata_text(self.REAL_ROW)
        assert "history: 3,1,4,2,7" in text
        assert "history: 0,0,0,0,0" not in text

    def test_none_and_blank_fields_do_not_crash(self):
        """None or whitespace-only speaker/party/subject become 'UNKNOWN' instead of raising."""
        row = dict(self.REAL_ROW, speaker=None, party_affiliation="  ", subject=None)
        text = build_metadata_text(row)
        assert "speaker: UNKNOWN" in text
        assert "party: UNKNOWN" in text
        assert "subject: UNKNOWN" in text

    def test_none_counts_become_zero(self):
        """A missing count is treated as 0 and must not shift the other positions."""
        row = dict(self.REAL_ROW, false_counts=None)
        assert "history: 3,0,4,2,7" in build_metadata_text(row)

    def test_legacy_credit_history_list_still_supported(self):
        """The older pre-combined `credit_history` list still works, zero-padded to 5 values."""
        row = {"statement": "x", "credit_history": [1, 2]}
        assert "history: 1,2,0,0,0" in build_metadata_text(row)


class TestTestSubset:
    """Test the stratified test-subset selection on a synthetic dataset."""

    @staticmethod
    def _fake_dataset(counts):
        """Dict stand-in for a dataset with counts[i] rows of class i (only "label" is read)."""
        labels = [i for i, n in enumerate(counts) for _ in range(n)]
        return {"label": labels}

    def test_exact_size_unique_and_sorted(self):
        """Rounding per class must not change the total: exactly 300, no duplicates, sorted."""
        ds = self._fake_dataset([92, 250, 214, 267, 249, 211])
        idx = get_test_indices(ds, subset_size=300)
        assert len(idx) == 300
        assert len(set(idx)) == 300
        assert idx == sorted(idx)

    def test_is_proportional_to_label_distribution(self):
        """Each class's share of the subset is within 1 of its exact proportional share."""
        counts = [92, 250, 214, 267, 249, 211]
        ds = self._fake_dataset(counts)
        idx = get_test_indices(ds, subset_size=300)
        total = sum(counts)
        for c, n in enumerate(counts):
            got = sum(1 for i in idx if ds["label"][i] == c)
            assert abs(got - n * 300 / total) <= 1

    def test_deterministic_and_does_not_touch_global_rng(self):
        """Same call gives the same subset (so all models share it) without reseeding or
        consuming numpy's global random state."""
        import numpy as np
        ds = self._fake_dataset([50, 50, 50, 50, 50, 50])
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        a = get_test_indices(ds, subset_size=60)
        assert (np.random.get_state()[1] == before).all()
        assert a == get_test_indices(ds, subset_size=60)

    def test_subset_larger_than_dataset_returns_all(self):
        """Requesting more rows than exist returns every index instead of failing."""
        ds = self._fake_dataset([2, 2, 2, 2, 2, 2])
        assert get_test_indices(ds, subset_size=100) == list(range(12))

    def test_invalid_subset_size(self):
        """A non-positive subset_size is rejected with ValueError."""
        with pytest.raises(ValueError):
            get_test_indices(self._fake_dataset([1, 1, 1, 1, 1, 1]), subset_size=0)


class TestDataLoading:
    """Test LIAR dataset loading (integration tests; need network or LIAR_TSV_DIR)."""

    def test_load_liar(self):
        """Test loading LIAR dataset."""
        train, val, test = load_liar()

        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

        # Check required fields
        for dataset in [train, val, test]:
            for row in dataset:
                assert "statement" in row
                assert "label" in row
                assert validate_label(row["label"])

    def test_real_rows_carry_nonzero_history_in_metadata(self):
        """Real LIAR rows must not all collapse to 'history: 0,0,0,0,0'."""
        train, _, _ = load_liar()
        texts = [build_metadata_text(train[i]) for i in range(200)]
        assert any("history: 0,0,0,0,0" not in t for t in texts)

    def test_get_test_indices(self):
        """Test test subset generation."""
        _, _, test = load_liar()

        indices = get_test_indices(test, subset_size=100)

        assert len(indices) == 100
        assert len(set(indices)) == 100  # All unique
        assert all(0 <= i < len(test) for i in indices)


class TestEvaluation:
    """Test evaluation metrics."""

    def test_evaluate_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 5]

        metrics = evaluate(y_true, y_pred)

        # Upper bound: a perfect classifier must score exactly 1.0 on both headline metrics.
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["n_samples"] == 6

    def test_evaluate_all_wrong(self):
        """Test evaluation with all wrong predictions."""
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]

        metrics = evaluate(y_true, y_pred)

        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] < 1.0
        assert metrics["n_samples"] == 4

    def test_evaluate_partial_correct(self):
        """Test evaluation with partial correctness."""
        y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        y_pred = [0, 1, 2, 3, 4, 5, 1, 1, 2, 3]  # One error

        metrics = evaluate(y_true, y_pred)

        assert 0 < metrics["accuracy"] < 1
        assert 0 < metrics["macro_f1"] < 1
        assert len(metrics["confusion_matrix"]) == 6
        assert metrics["n_samples"] == 10

    def test_evaluate_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        y_true = [0, 1, 2]
        y_pred = [0, 1]

        with pytest.raises(ValueError):
            evaluate(y_true, y_pred)

    def test_evaluate_empty(self):
        """Test that empty predictions raise error."""
        with pytest.raises(ValueError):
            evaluate([], [])

    def test_evaluate_confusion_matrix(self):
        """Test that confusion matrix is 6x6."""
        y_true = list(range(6)) * 2
        y_pred = list(range(6)) * 2

        metrics = evaluate(y_true, y_pred)

        cm = metrics["confusion_matrix"]
        assert len(cm) == 6
        assert all(len(row) == 6 for row in cm)

    def test_build_results_dict(self):
        """Test building standardized results dictionary."""
        y_true = [0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 5]
        metrics = evaluate(y_true, y_pred)

        results = build_results_dict(
            model_name="test_bert",
            metrics=metrics,
            params=110000000,
            train_minutes=5.0,
            gpu="T4",
            avg_latency_ms=50.0,
            est_cost_per_1k=0.10,
            notes="Test run"
        )

        assert results["model"] == "test_bert"
        assert results["accuracy"] == 1.0
        assert results["params"] == 110000000
        assert results["train_minutes"] == 5.0
        assert results["gpu"] == "T4"
        assert "confusion_matrix" in results
        assert "per_class_metrics" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
