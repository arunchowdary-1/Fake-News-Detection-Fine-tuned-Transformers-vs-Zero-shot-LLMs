"""
Unit tests for common module (data loader, config).
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


class TestDataLoading:
    """Test LIAR dataset loading (integration tests)."""

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

    def test_get_test_indices(self):
        """Test test subset generation."""
        _, _, test = load_liar()

        indices = get_test_indices(test, subset_size=100)

        assert len(indices) == 100
        assert len(set(indices)) == 100  # All unique
        assert all(0 <= i < len(test) for i in indices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
