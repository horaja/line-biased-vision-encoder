"""
Training callbacks for the SelectiveMagnoViT model.

Callbacks provide hooks into the training process for logging, checkpointing,
and other custom behavior. This is kept simple for now since the Trainer class
already has most of this functionality built in.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base class for training callbacks.

    Callbacks can hook into various points of the training process.
    """

    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, trainer):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch_idx: int, trainer):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        """Called at the end of each batch."""
        pass


class MetricsLogger(Callback):
    """
    Callback for logging metrics during training.

    This is a simple callback that logs metrics at the end of each epoch.
    """

    def __init__(self, log_every_n_epochs: int = 1):
        """
        Args:
            log_every_n_epochs: Log metrics every n epochs
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Log metrics at the end of epoch."""
        if (epoch + 1) % self.log_every_n_epochs == 0:
            logger.info(f"Epoch {epoch + 1}: {metrics}")


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on validation metrics.

    Note: The Trainer class already has early stopping built in,
    so this is mainly for demonstration purposes.
    """

    def __init__(
        self,
        monitor: str = 'val_accuracy',
        patience: int = 10,
        mode: str = 'max',
        min_delta: float = 0.0
    ):
        """
        Args:
            monitor: Metric to monitor (e.g., 'val_accuracy', 'val_loss')
            patience: Number of epochs with no improvement before stopping
            mode: 'max' for metrics that should be maximized, 'min' for minimized
            min_delta: Minimum change to qualify as an improvement
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Check if training should stop."""
        if self.monitor not in metrics:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return

        current_value = metrics[self.monitor]

        if self.best_value is None:
            self.best_value = current_value
        else:
            if self._is_improvement(current_value):
                self.best_value = current_value
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    # Note: Actual stopping would need to be implemented in the Trainer

    def _is_improvement(self, current_value: float) -> bool:
        """Check if current value is an improvement over best value."""
        if self.mode == 'max':
            return current_value > (self.best_value + self.min_delta)
        else:  # mode == 'min'
            return current_value < (self.best_value - self.min_delta)


class CheckpointCallback(Callback):
    """
    Callback for saving checkpoints during training.

    Note: The Trainer class already handles checkpointing,
    so this is mainly for custom checkpoint strategies.
    """

    def __init__(
        self,
        save_every_n_epochs: int = 5,
        save_best_only: bool = False,
        monitor: str = 'val_accuracy'
    ):
        """
        Args:
            save_every_n_epochs: Save checkpoint every n epochs
            save_best_only: Only save when model improves
            monitor: Metric to monitor for best model
        """
        self.save_every_n_epochs = save_every_n_epochs
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.best_value = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Save checkpoint if conditions are met."""
        should_save = False

        if not self.save_best_only:
            # Save every n epochs
            if (epoch + 1) % self.save_every_n_epochs == 0:
                should_save = True

        if self.monitor in metrics:
            current_value = metrics[self.monitor]
            if self.best_value is None or current_value > self.best_value:
                self.best_value = current_value
                if self.save_best_only:
                    should_save = True

        if should_save:
            logger.info(f"Saving checkpoint at epoch {epoch + 1}")
            # Actual saving would be handled by the Trainer


class LearningRateLogger(Callback):
    """
    Callback for logging learning rate during training.
    """

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Log current learning rate."""
        if hasattr(trainer, 'optimizer'):
            lr = trainer.optimizer.param_groups[0]['lr']
            logger.debug(f"Epoch {epoch + 1}: Learning rate = {lr:.2e}")


class CallbackList:
    """
    Container for managing multiple callbacks.

    Usage:
        callbacks = CallbackList([
            MetricsLogger(),
            EarlyStoppingCallback(patience=10)
        ])
    """

    def __init__(self, callbacks: list = None):
        """
        Args:
            callbacks: List of Callback instances
        """
        self.callbacks = callbacks if callbacks is not None else []

    def on_train_begin(self, trainer):
        """Call all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer):
        """Call all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer):
        """Call all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Call all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, trainer)

    def on_batch_begin(self, batch_idx: int, trainer):
        """Call all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, trainer)

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        """Call all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, trainer)

    def add(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def remove(self, callback: Callback):
        """Remove a callback from the list."""
        self.callbacks.remove(callback)
