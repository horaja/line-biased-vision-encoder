"""
Model evaluation utilities for SelectiveMagnoViT.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..utils.metrics import (
    compute_accuracy,
    compute_topk_accuracy,
    compute_confusion_matrix,
    compute_per_class_accuracy,
    compute_precision_recall_f1,
    compute_gflops
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates a trained SelectiveMagnoViT model.

    Features:
    - Compute accuracy, top-k accuracy, and other metrics
    - Generate confusion matrix
    - Per-class performance analysis
    - Save evaluation results
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
            class_names: List of class names (for reporting)
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_confusion: bool = True,
        compute_topk: bool = True,
        k: int = 5
    ) -> Dict:
        """
        Evaluate the model on a dataset.

        Args:
            dataloader: DataLoader for the evaluation dataset
            compute_confusion: Whether to compute confusion matrix
            compute_topk: Whether to compute top-k accuracy
            k: k for top-k accuracy

        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting evaluation...")

        # Metrics storage
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_outputs = []

        criterion = nn.CrossEntropyLoss()

        # Calculate gflops
        gflops = compute_gflops(self.model, self.device)
        logger.info(f"Model Efficiency: {gflops:.4f} GFLOPs per image")

        # Iterate through dataset
        progress_bar = tqdm(dataloader, desc="Evaluating")
        for batch in progress_bar:
            # Move data to device
            color_images = batch['color_image'].to(self.device)
            line_drawings = batch['line_drawing'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = self.model(color_images, line_drawings)
            loss = criterion(outputs, labels)

            # Store results
            total_loss += loss.item() * color_images.size(0)
            total_samples += color_images.size(0)

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_outputs.append(outputs.cpu())

            # Update progress
            current_acc = (torch.tensor(all_predictions) == torch.tensor(all_targets)).float().mean().item()
            progress_bar.set_postfix({'accuracy': f"{current_acc:.4f}"})

        # Compute metrics
        avg_loss = total_loss / total_samples
        accuracy = (torch.tensor(all_predictions) == torch.tensor(all_targets)).float().mean().item()

        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total_samples,
            'gflops': gflops
        }

        # Top-k accuracy
        if compute_topk:
            all_outputs_tensor = torch.cat(all_outputs, dim=0)
            all_targets_tensor = torch.tensor(all_targets)
            topk_acc = compute_topk_accuracy(all_outputs_tensor, all_targets_tensor, k=k)
            results[f'top{k}_accuracy'] = topk_acc
            logger.info(f"Top-{k} Accuracy: {topk_acc:.4f}")

        # Confusion matrix and per-class metrics
        if compute_confusion:
            num_classes = len(set(all_targets))
            confusion = compute_confusion_matrix(all_predictions, all_targets, num_classes)
            per_class_acc = compute_per_class_accuracy(confusion)

            results['confusion_matrix'] = confusion.tolist()
            results['per_class_accuracy'] = per_class_acc.tolist()

            # Per-class precision, recall, F1
            per_class_metrics = []
            for class_idx in range(num_classes):
                precision, recall, f1 = compute_precision_recall_f1(confusion, class_idx)
                class_name = self.class_names[class_idx] if self.class_names else f"Class_{class_idx}"
                per_class_metrics.append({
                    'class': class_name,
                    'accuracy': per_class_acc[class_idx],
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

            results['per_class_metrics'] = per_class_metrics

            # Log per-class results
            logger.info("\nPer-class Performance:")
            logger.info("-" * 80)
            logger.info(f"{'Class':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
            logger.info("-" * 80)
            for metrics in per_class_metrics:
                logger.info(
                    f"{metrics['class']:<20} "
                    f"{metrics['accuracy']:>10.4f} "
                    f"{metrics['precision']:>10.4f} "
                    f"{metrics['recall']:>10.4f} "
                    f"{metrics['f1_score']:>10.4f}"
                )
            logger.info("-" * 80)

        # Overall results
        logger.info(f"\nOverall Results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Total Samples: {total_samples}")

        return results

    def save_results(self, results: Dict, output_path: str):
        """
        Save evaluation results to a JSON file.

        Args:
            results: Results dictionary from evaluate()
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {output_path}")

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        return_probabilities: bool = False
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """
        Make predictions on a dataset.

        Args:
            dataloader: DataLoader for the dataset
            return_probabilities: Whether to return class probabilities

        Returns:
            Tuple of (predictions, probabilities) where probabilities is None
            if return_probabilities is False
        """
        all_predictions = []
        all_probabilities = [] if return_probabilities else None

        progress_bar = tqdm(dataloader, desc="Predicting")
        for batch in progress_bar:
            color_images = batch['color_image'].to(self.device)
            line_drawings = batch['line_drawing'].to(self.device)

            outputs = self.model(color_images, line_drawings)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy().tolist())

            if return_probabilities:
                probs = torch.softmax(outputs, dim=1)
                all_probabilities.append(probs.cpu().numpy())

        if return_probabilities:
            all_probabilities = np.concatenate(all_probabilities, axis=0)

        return all_predictions, all_probabilities

    @torch.no_grad()
    def evaluate_patch_selection(
        self,
        dataloader: DataLoader,
        num_samples: int = 100
    ) -> Dict:
        """
        Analyze patch selection behavior.

        Args:
            dataloader: DataLoader for the dataset
            num_samples: Number of samples to analyze

        Returns:
            Dictionary with patch selection statistics
        """
        patch_counts = []
        patch_percentages = []

        samples_processed = 0
        for batch in tqdm(dataloader, desc="Analyzing patch selection"):
            if samples_processed >= num_samples:
                break

            line_drawings = batch['line_drawing'].to(self.device)

            # Get selected patch indices
            selected_indices = self.model.get_selected_patch_indices(line_drawings)

            batch_size = selected_indices.size(0)
            num_selected = selected_indices.size(1)
            total_patches = self.model.num_patches

            patch_counts.extend([num_selected] * batch_size)
            patch_percentages.extend([num_selected / total_patches] * batch_size)

            samples_processed += batch_size

        results = {
            'avg_patches_selected': np.mean(patch_counts),
            'std_patches_selected': np.std(patch_counts),
            'avg_patch_percentage': np.mean(patch_percentages),
            'total_patches': self.model.num_patches,
            'samples_analyzed': len(patch_counts)
        }

        logger.info("\nPatch Selection Statistics:")
        logger.info(f"  Avg patches selected: {results['avg_patches_selected']:.2f} +/- {results['std_patches_selected']:.2f}")
        logger.info(f"  Avg percentage: {results['avg_patch_percentage']:.2%}")
        logger.info(f"  Total patches: {results['total_patches']}")

        return results
