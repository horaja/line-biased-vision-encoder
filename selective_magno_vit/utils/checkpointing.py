"""
Checkpoint management utilities for saving and loading model states.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.

    Handles:
    - Saving full training state (model, optimizer, scheduler, metrics)
    - Loading checkpoints for resuming training
    - Keeping track of best models
    - Automatic checkpoint cleanup to save disk space
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
            save_best: Whether to save a separate best model checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = None
        self.checkpoint_history = []

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save a checkpoint with full training state.

        Args:
            model: Model to save
            optimizer: Optimizer state (optional)
            scheduler: Learning rate scheduler state (optional)
            epoch: Current epoch number
            metrics: Dictionary of metrics (e.g., {'val_acc': 0.95, 'val_loss': 0.1})
            extra_info: Any additional information to save
            is_best: Whether this is the best model so far
            filename: Custom filename (if None, uses epoch number)

        Returns:
            Path to the saved checkpoint
        """
        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if extra_info is not None:
            checkpoint['extra_info'] = extra_info

        # Determine filename
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"

        checkpoint_path = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Track checkpoint
        self.checkpoint_history.append(checkpoint_path)

        # Save best model separately if requested
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

            # Also save just the model weights for easy loading
            best_weights_path = self.checkpoint_dir / "best_model_weights.pth"
            torch.save(model.state_dict(), best_weights_path)

        # Clean up old checkpoints if needed
        if self.max_checkpoints > 0 and len(self.checkpoint_history) > self.max_checkpoints:
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore training state.

        Args:
            checkpoint_path: Path to the checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            device: Device to map tensors to (optional)

        Returns:
            Dictionary containing checkpoint information (epoch, metrics, etc.)
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        if device is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path)

        # Restore model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model weights from {checkpoint_path}")

        # Restore optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Restored optimizer state")

        # Restore scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Restored scheduler state")

        # Return checkpoint info
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'extra_info': checkpoint.get('extra_info', {}),
            'timestamp': checkpoint.get('timestamp', None)
        }

    def load_best_model(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load the best saved model.

        Args:
            model: Model to load weights into
            device: Device to map tensors to (optional)

        Returns:
            Dictionary containing checkpoint information
        """
        best_path = self.checkpoint_dir / "best_model.pth"

        if not best_path.exists():
            raise FileNotFoundError(f"Best model not found: {best_path}")

        return self.load_checkpoint(best_path, model, device=device)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        # Keep the most recent max_checkpoints
        checkpoints_to_remove = self.checkpoint_history[:-self.max_checkpoints]

        for checkpoint_path in checkpoints_to_remove:
            # Don't remove best model
            if checkpoint_path.name == "best_model.pth":
                continue

            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint_path}")

        # Update history
        self.checkpoint_history = self.checkpoint_history[-self.max_checkpoints:]

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the most recent checkpoint."""
        if not self.checkpoint_history:
            return None
        return self.checkpoint_history[-1]

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return list(self.checkpoint_dir.glob("checkpoint_*.pth"))


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    **kwargs
):
    """
    Simple function to save a checkpoint (convenience wrapper).

    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
        epoch: Current epoch
        **kwargs: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    checkpoint.update(kwargs)

    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Simple function to load a checkpoint (convenience wrapper).

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
        device: Device to map to (optional)

    Returns:
        Dictionary with checkpoint info
    """
    if device is not None:
        checkpoint = torch.load(filepath, map_location=device)
    else:
        checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logger.info(f"Loaded checkpoint from {filepath}")

    return checkpoint
