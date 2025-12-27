"""
Training loop implementation for SelectiveMagnoViT.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils.metrics import MetricTracker, compute_accuracy, format_metrics
from ..utils.checkpointing import CheckpointManager

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles training and validation of the SelectiveMagnoViT model.

    Features:
    - Training and validation loops with progress bars
    - Automatic checkpointing and best model tracking
    - TensorBoard logging
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object with training parameters
            device: Device to train on (cuda/cpu)
            logger: Logger instance (if None, uses module logger)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Setup optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get('training.learning_rate', 1e-4),
            weight_decay=config.get('training.weight_decay', 0.01)
        )

        # Setup learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('training.epochs', 100),
            eta_min=1e-6
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Setup checkpoint manager
        checkpoint_dir = config.get('output.checkpoint_dir', 'checkpoints')
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=5,
            save_best=True
        )

        # Setup TensorBoard
        tensorboard_dir = config.get('output.tensorboard_dir')
        if tensorboard_dir:
            self.writer = SummaryWriter(tensorboard_dir)
            self.logger.info(f"TensorBoard logging to {tensorboard_dir}")
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.patience = config.get('training.patience', 20)

        # Metric trackers
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            color_images = batch['color_image'].to(self.device)
            line_drawings = batch['line_drawing'].to(self.device)
            labels = batch['label'].to(self.device)

            batch_size = color_images.size(0)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(color_images, line_drawings)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute metrics
            accuracy = compute_accuracy(outputs, labels)

            # Update metrics
            self.train_metrics.update('loss', loss.item(), batch_size)
            self.train_metrics.update('accuracy', accuracy, batch_size)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.4f}"
            })

        # Get average metrics
        metrics = self.train_metrics.get_all_averages()
        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]  ",
            leave=False
        )

        with torch.no_grad():
            for batch in progress_bar:
                # Move data to device
                color_images = batch['color_image'].to(self.device)
                line_drawings = batch['line_drawing'].to(self.device)
                labels = batch['label'].to(self.device)
                
                batch_size = color_images.size(0)

                # Forward pass
                outputs = self.model(color_images, line_drawings)
                loss = self.criterion(outputs, labels)

                # Compute metrics
                accuracy = compute_accuracy(outputs, labels)

                # Update metrics
                self.val_metrics.update('loss', loss.item(), batch_size)
                self.val_metrics.update('accuracy', accuracy, batch_size)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{accuracy:.4f}"
                })

        # Get average metrics
        metrics = self.val_metrics.get_all_averages()
        return metrics

    def train(self) -> float:
        """
        Full training loop.

        Returns:
            Best validation accuracy achieved
        """
        num_epochs = self.config.get('training.epochs', 100)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("TRAINING CONFIGURATION")
        self.logger.info("=" * 70)
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Batch size: {self.config.get('training.batch_size')}")
        self.logger.info(f"Learning rate: {self.config.get('training.learning_rate')}")
        self.logger.info(f"Patch percentage: {self.config.get('model.patch_percentage')}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info("=" * 70 + "\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Learning_rate', current_lr, epoch)

            # Print progress
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch + 1:03d}/{num_epochs:03d} | "
                f"Train: {format_metrics(train_metrics)} | "
                f"Val: {format_metrics(val_metrics)} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Check if best model
            val_accuracy = val_metrics['accuracy']
            is_best = val_accuracy > self.best_val_accuracy

            if is_best:
                self.best_val_accuracy = val_accuracy
                self.patience_counter = 0
                self.logger.info(f"  -> New best model! Val Accuracy: {val_accuracy:.4f}")
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=is_best,
                    extra_info={'train_metrics': train_metrics}
                )
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                self.logger.info(f"No improvement for {self.patience} epochs.")
                break

        # Training complete
        self.logger.info("\n" + "=" * 70)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        self.logger.info(f"Total epochs: {self.current_epoch + 1}")
        self.logger.info("=" * 70)

        if self.writer:
            self.writer.close()

        return self.best_val_accuracy

    def save_checkpoint(self, filename: str):
        """
        Save a checkpoint manually.

        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            metrics={'best_val_accuracy': self.best_val_accuracy},
            filename=filename
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint to resume training.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )

        self.current_epoch = info.get('epoch', 0)

        metrics = info.get('metrics', {})
        if 'accuracy' in metrics:
            self.best_val_accuracy = metrics['accuracy']

        self.logger.info(f"Resumed from epoch {self.current_epoch}")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
