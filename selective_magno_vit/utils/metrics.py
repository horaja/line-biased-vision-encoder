"""
Metrics computation utilities for model evaluation.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from collections import defaultdict


class MetricTracker:
    """
    Tracks and aggregates metrics during training/evaluation.

    Usage:
        tracker = MetricTracker()
        for batch in dataloader:
            loss, acc = compute_metrics(batch)
            tracker.update('loss', loss, batch_size)
            tracker.update('accuracy', acc, batch_size)

        avg_loss = tracker.avg('loss')
        avg_acc = tracker.avg('accuracy')
    """

    def __init__(self):
        self.metrics = defaultdict(lambda: {'sum': 0.0, 'count': 0})
        self.history = defaultdict(list)

    def update(self, name: str, value: float, count: int = 1):
        """
        Update a metric with a new value.

        Args:
            name: Name of the metric
            value: Value to add (can be a batch average)
            count: Number of samples this value represents (e.g., batch size)
        """
        self.metrics[name]['sum'] += value * count
        self.metrics[name]['count'] += count

    def avg(self, name: str) -> float:
        """Get the average value of a metric."""
        if self.metrics[name]['count'] == 0:
            return 0.0
        return self.metrics[name]['sum'] / self.metrics[name]['count']

    def get_all_averages(self) -> dict:
        """Get all metric averages as a dictionary."""
        return {name: self.avg(name) for name in self.metrics.keys()}

    def save_to_history(self):
        """Save current averages to history and reset."""
        for name in self.metrics.keys():
            self.history[name].append(self.avg(name))
        self.reset()

    def reset(self):
        """Reset all metrics to zero."""
        self.metrics.clear()

    def __repr__(self):
        return f"MetricTracker({self.get_all_averages()})"


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        outputs: Model predictions of shape (B, num_classes) - logits or probabilities
        targets: Ground truth labels of shape (B,)

    Returns:
        Accuracy as a float between 0 and 1
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def compute_topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """
    Compute top-k accuracy.

    Args:
        outputs: Model predictions of shape (B, num_classes)
        targets: Ground truth labels of shape (B,)
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy as a float between 0 and 1
    """
    _, topk_predictions = torch.topk(outputs, k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(topk_predictions)
    correct = (topk_predictions == targets_expanded).any(dim=1).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def compute_confusion_matrix(
    all_predictions: List[int],
    all_targets: List[int],
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        all_predictions: List of predicted class indices
        all_targets: List of ground truth class indices
        num_classes: Total number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes) where
        element [i, j] is the number of times class i was predicted as class j
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, target in zip(all_predictions, all_targets):
        confusion_matrix[target, pred] += 1

    return confusion_matrix


def compute_per_class_accuracy(confusion_matrix: np.ndarray) -> np.ndarray:
    """
    Compute per-class accuracy from confusion matrix.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)

    Returns:
        Per-class accuracy array of shape (num_classes,)
    """
    per_class_correct = np.diag(confusion_matrix)
    per_class_total = confusion_matrix.sum(axis=1)

    # Avoid division by zero
    per_class_accuracy = np.zeros(len(per_class_correct))
    mask = per_class_total > 0
    per_class_accuracy[mask] = per_class_correct[mask] / per_class_total[mask]

    return per_class_accuracy


def compute_precision_recall_f1(
    confusion_matrix: np.ndarray,
    class_idx: int
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for a specific class.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes)
        class_idx: Index of the class to compute metrics for

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    true_positives = confusion_matrix[class_idx, class_idx]
    false_positives = confusion_matrix[:, class_idx].sum() - true_positives
    false_negatives = confusion_matrix[class_idx, :].sum() - true_positives

    # Compute precision
    precision = 0.0
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)

    # Compute recall
    recall = 0.0
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)

    # Compute F1 score
    f1_score = 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking running averages during training.
    """

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update with a new value.

        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"

    def __repr__(self):
        return f"AverageMeter(name={self.name}, avg={self.avg:.4f})"


def format_metrics(metrics: dict, precision: int = 4) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metric names and values
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    formatted = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            formatted.append(f"{name}: {value:.{precision}f}")
        else:
            formatted.append(f"{name}: {value}")

    return " | ".join(formatted)
