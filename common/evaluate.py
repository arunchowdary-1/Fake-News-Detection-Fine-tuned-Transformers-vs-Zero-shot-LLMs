"""
Shared evaluation metrics and results schema for model comparison.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)

from common.config import IDX_TO_LABEL, RESULTS_DIR


def evaluate(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    Evaluate predictions against ground truth.

    Args:
        y_true: Ground truth labels (0-5)
        y_pred: Predicted labels (0-5)

    Returns:
        Dictionary with accuracy, macro_f1, per-class metrics, confusion matrix
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) have different lengths")

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions")

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(6))

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        labels=range(6),
        target_names=[IDX_TO_LABEL[i] for i in range(6)],
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": report,
        "n_samples": len(y_true),
    }


def save_predictions_csv(
    ids: List[int],
    y_true: List[int],
    y_pred: List[int],
    latencies_ms: List[float],
    model_name: str,
    output_dir: str = RESULTS_DIR
) -> str:
    """
    Save predictions to CSV for model comparison.

    Args:
        ids: Sample IDs/indices
        y_true: Ground truth labels
        y_pred: Predicted labels
        latencies_ms: Latency per sample (milliseconds)
        model_name: Name of the model (e.g., 'lstm', 'bert_text', 'llm')
        output_dir: Directory to save to

    Returns:
        Path to saved CSV
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, f"{model_name}_preds.csv")

    with open(filepath, "w") as f:
        f.write("id,y_true,y_pred,latency_ms\n")
        for idx, true_label, pred_label, latency in zip(ids, y_true, y_pred, latencies_ms):
            f.write(f"{idx},{true_label},{pred_label},{latency:.2f}\n")

    print(f"Saved predictions to {filepath}")
    return filepath


def save_results_json(
    results: Dict,
    model_name: str,
    output_dir: str = RESULTS_DIR
) -> str:
    """
    Save evaluation results to JSON.

    Args:
        results: Results dictionary with accuracy, macro_f1, etc.
        model_name: Name of the model
        output_dir: Directory to save to

    Returns:
        Path to saved JSON
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, f"{model_name}.json")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {filepath}")
    return filepath


def build_results_dict(
    model_name: str,
    metrics: Dict,
    params: int,
    train_minutes: float,
    gpu: str,
    avg_latency_ms: float,
    est_cost_per_1k: float,
    cost_assumption: str = "T4: $0.35/hour",
    invalid_outputs: int = 0,
    notes: str = ""
) -> Dict:
    """
    Build a standardized results dictionary.

    Args:
        model_name: Model identifier (e.g., 'bert_text', 'llm_qwen')
        metrics: Evaluation metrics dict from evaluate()
        params: Number of model parameters
        train_minutes: Training time in minutes (0 for inference-only models)
        gpu: GPU type used (e.g., 'T4', 'CPU', 'None')
        avg_latency_ms: Average inference latency per sample
        est_cost_per_1k: Estimated cost per 1,000 predictions (USD)
        cost_assumption: Assumption about GPU pricing
        invalid_outputs: Number of samples with invalid/failed predictions
        notes: Additional notes

    Returns:
        Standardized results dictionary
    """
    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "params": params,
        "train_minutes": train_minutes,
        "gpu": gpu,
        "avg_latency_ms": avg_latency_ms,
        "est_cost_per_1k": est_cost_per_1k,
        "cost_assumption": cost_assumption,
        "n_eval": metrics["n_samples"],
        "invalid_outputs": invalid_outputs,
        "notes": notes,
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_metrics": metrics["per_class_metrics"],
    }


def load_results_json(model_name: str, input_dir: str = RESULTS_DIR) -> Dict:
    """
    Load evaluation results from JSON.

    Args:
        model_name: Name of the model
        input_dir: Directory to load from

    Returns:
        Results dictionary
    """
    filepath = os.path.join(input_dir, f"{model_name}.json")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath, "r") as f:
        results = json.load(f)

    return results


def compare_models(model_names: List[str], input_dir: str = RESULTS_DIR) -> Dict:
    """
    Load and compare results from multiple models.

    Args:
        model_names: List of model names to load
        input_dir: Directory to load from

    Returns:
        Dictionary with all results for easy comparison
    """
    all_results = {}

    for model_name in model_names:
        try:
            all_results[model_name] = load_results_json(model_name, input_dir)
        except FileNotFoundError:
            print(f"Warning: Results not found for {model_name}")

    return all_results


def print_comparison_table(model_names: List[str], input_dir: str = RESULTS_DIR) -> None:
    """
    Print a formatted comparison table of model results.

    Args:
        model_names: List of model names to compare
        input_dir: Directory to load from
    """
    all_results = compare_models(model_names, input_dir)

    if not all_results:
        print("No results to compare")
        return

    print("\n" + "=" * 120)
    print("MODEL COMPARISON TABLE".center(120))
    print("=" * 120)

    # Header
    print(f"{'Model':<15} {'Accuracy':<12} {'Macro F1':<12} {'Params':<15} {'Train (min)':<12} {'Latency (ms)':<13} {'Cost/1k':<10}")
    print("-" * 120)

    # Rows
    for model_name in model_names:
        if model_name in all_results:
            r = all_results[model_name]
            print(f"{model_name:<15} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['params']:<15,d} {r['train_minutes']:<12.2f} {r['avg_latency_ms']:<13.1f} ${r['est_cost_per_1k']:<9.3f}")

    print("=" * 120 + "\n")


if __name__ == "__main__":
    # Test the evaluate function
    print("Testing evaluation metrics...")

    y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 4, 5, 1, 1, 2, 3]  # One error at index 6

    metrics = evaluate(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")

    # Test results dict building
    results = build_results_dict(
        model_name="test_model",
        metrics=metrics,
        params=1000000,
        train_minutes=5.0,
        gpu="T4",
        avg_latency_ms=50.5,
        est_cost_per_1k=0.10,
        notes="Test run"
    )

    print(f"\nResults dict: {json.dumps(results, indent=2, default=str)[:500]}...")
