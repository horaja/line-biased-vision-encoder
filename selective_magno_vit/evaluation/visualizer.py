"""
Visualization utilities for SelectiveMagnoViT.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Tuple, Optional
import seaborn as sns


class PatchSelectionVisualizer:
    """
    Visualizes patch selection behavior of the model.
    """

    def __init__(self, model, device: torch.device):
        """
        Args:
            model: SelectiveMagnoViT model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def visualize_patch_selection(
        self,
        color_image: torch.Tensor,
        line_drawing: Optional[torch.Tensor], # Changed type hint
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize which patches are selected for a given image.

        Args:
            color_image: Color image tensor (1, 3, H, W)
            line_drawing: Line drawing tensor (1, 1, H, W) or None
            save_path: Path to save figure (optional)
            show: Whether to show the figure
        """
        # Get model predictions
        color_image = color_image.to(self.device)
        
        # CHANGED: Handle None line_drawing
        if line_drawing is not None:
            line_drawing = line_drawing.to(self.device)
            # Get importance scores only if line drawing exists
            patch_scores, patch_centers = self.model.scorer(line_drawing)
            scores_np = patch_scores[0].cpu().numpy()
            centers_np = patch_centers[0].cpu().numpy()
            line_np = line_drawing[0, 0].cpu().numpy()
        else:
            scores_np = None
            centers_np = None
            line_np = None

        # Get selected patch indices (works with None if random/supported, passing color_image)
        selected_indices = self.model.get_selected_patch_indices(line_drawing, color_image)

        # Convert to numpy for plotting
        color_np = color_image[0].cpu().permute(1, 2, 0).numpy()
        selected_indices_np = selected_indices[0].cpu().numpy()

        # Denormalize color image for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        color_np = std * color_np + mean
        color_np = np.clip(color_np, 0, 1)

        # Calculate patch grid
        color_img_size = color_np.shape[0]
        color_patch_size = self.model.color_patch_size
        num_color_patches_per_side = color_img_size // color_patch_size

        # Create figure
        fig = plt.figure(figsize=(16, 4))
        gs = GridSpec(1, 4, figure=fig, wspace=0.3)

        # 1. Original color image
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(color_np)
        ax1.set_title("Color Image")
        ax1.axis('off')

        # 2. Line drawing (Handle None)
        ax2 = fig.add_subplot(gs[1])
        if line_np is not None:
            ax2.imshow(line_np, cmap='gray')
            ax2.set_title("Line Drawing")
        else:
            ax2.text(0.5, 0.5, "Not Available", ha='center', va='center')
            ax2.set_title("Line Drawing (None)")
        ax2.axis('off')

        # 3. Patch importance scores (Handle None)
        ax3 = fig.add_subplot(gs[2])
        if scores_np is not None:
            line_patch_size = self.model.ld_patch_size
            line_img_size = 256 if line_np is None else line_np.shape[0] # Fallback size
            if line_np is not None:
                num_line_patches_per_side = line_img_size // line_patch_size
                scores_2d = scores_np.reshape(num_line_patches_per_side, num_line_patches_per_side)
                im = ax3.imshow(scores_2d, cmap='hot', interpolation='nearest')
                plt.colorbar(im, ax=ax3, fraction=0.046)
            ax3.set_title("Patch Importance Scores")
        else:
            ax3.text(0.5, 0.5, "Random/None", ha='center', va='center')
            ax3.set_title("Importance Scores")
        ax3.axis('off')

        # 4. Selected patches overlay
        ax4 = fig.add_subplot(gs[3])
        ax4.imshow(color_np)

        # Draw all patch boundaries (light)
        for i in range(num_color_patches_per_side + 1):
            ax4.axhline(y=i * color_patch_size - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            ax4.axvline(x=i * color_patch_size - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        # Highlight selected patches
        for idx in selected_indices_np:
            patch_y = (idx // num_color_patches_per_side) * color_patch_size
            patch_x = (idx % num_color_patches_per_side) * color_patch_size

            rect = patches.Rectangle(
                (patch_x - 0.5, patch_y - 0.5),
                color_patch_size,
                color_patch_size,
                linewidth=2,
                edgecolor='lime',
                facecolor='none'
            )
            ax4.add_patch(rect)

        # COG visualization (only if we have centers)
        if centers_np is not None:
            # cog_np is [y, x] in normalized coordinates [0, 1]
            cog_y_norm, cog_x_norm = centers_np
            
            # Map to Color Image patch grid indices
            cog_patch_row = int(cog_y_norm * num_color_patches_per_side)
            cog_patch_col = int(cog_x_norm * num_color_patches_per_side)
            
            # Clamp to ensure it stays within bounds
            cog_patch_row = max(0, min(cog_patch_row, num_color_patches_per_side - 1))
            cog_patch_col = max(0, min(cog_patch_col, num_color_patches_per_side - 1))
            
            # Calculate pixel coordinates for the CoG rect
            cog_rect_y = cog_patch_row * color_patch_size
            cog_rect_x = cog_patch_col * color_patch_size
            
            cog_rect = patches.Rectangle(
                (cog_rect_x - 0.5, cog_rect_y - 0.5),
                color_patch_size,
                color_patch_size,
                linewidth=3,        # Thicker
                edgecolor='red',    # Distinct color (Red)
                facecolor='none',
                linestyle='--'      # Dashed style
            )
            ax4.add_patch(cog_rect)

        num_selected = len(selected_indices_np)
        total_patches = num_color_patches_per_side ** 2
        ax4.set_title(f"Selected Patches ({num_selected}/{total_patches})")
        ax4.axis('off')

        plt.suptitle(f"Patch Selection Visualization (Patch Percentage: {self.model.patch_percentage:.2%})",
                     fontsize=14, y=1.02)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    @torch.no_grad()
    def visualize_multiple_samples(
        self,
        dataloader,
        num_samples: int = 5,
        save_dir: Optional[str] = None
    ):
        """
        Visualize patch selection for multiple samples.

        Args:
            dataloader: DataLoader with samples
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        samples_visualized = 0
        for batch in dataloader:
            if samples_visualized >= num_samples:
                break
            
            color_images = batch['color_image']
            
            # CHANGED: Handle None line drawings (Dataset level)
            line_drawings = batch['line_drawing']
            # line_drawings is None if the dataset returns None for all items
            
            labels = batch['label']

            batch_size = color_images.size(0)
            for i in range(min(batch_size, num_samples - samples_visualized)):
                save_path = None
                if save_dir:
                    save_path = save_dir / f"patch_selection_sample_{samples_visualized:03d}.png"

                # CHANGED: Pass None if batch line_drawings is None
                ld_sample = line_drawings[i:i+1] if line_drawings is not None else None

                self.visualize_patch_selection(
                    color_images[i:i+1],
                    ld_sample,
                    save_path=save_path,
                    show=False
                )

                samples_visualized += 1
                if samples_visualized >= num_samples:
                    break


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
    normalize: bool = False
):
    """
    Plot a confusion matrix.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save figure (optional)
        show: Whether to show the figure
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved confusion matrix to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save figure (optional)
        show: Whether to show the figure
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved training curves to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_per_class_performance(
    class_names: List[str],
    accuracies: List[float],
    f1_scores: List[float],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot per-class performance metrics.

    Args:
        class_names: List of class names
        accuracies: List of per-class accuracies
        f1_scores: List of per-class F1 scores
        save_path: Path to save figure (optional)
        show: Whether to show the figure
    """
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved per-class performance to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()