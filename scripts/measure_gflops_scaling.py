import torch
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from selective_magno_vit.models.selective_vit import SelectiveMagnoViT
from selective_magno_vit.utils.metrics import compute_gflops

def main():
    # --- CONFIGURATION ---
    # Model: ViT-Large (to see significant scaling)
    VIT_MODEL_NAME = "vit_large_patch16_224.augreg_in21k" 
    
    # Input Dimensions
    COLOR_IMG_SIZE = 256
    COLOR_PATCH_SIZE = 16
    
    # Percentages: 10% to 100%
    patch_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Measuring GFLOPS on {device}...")
    print(f"Model: {VIT_MODEL_NAME} | Input: {COLOR_IMG_SIZE}x{COLOR_IMG_SIZE}")
    
    # --- MEASUREMENT LOOP ---
    results_gflops = []
    
    for pp in patch_percentages:
        print(f"  Measuring for {pp*100:.0f}% patches...")
        
        # Initialize model (Random weights are fine for architecture analysis)
        model = SelectiveMagnoViT(
            patch_percentage=pp,
            num_classes=1000, 
            color_img_size=COLOR_IMG_SIZE,
            color_patch_size=COLOR_PATCH_SIZE,
            ld_img_size=64,
            ld_patch_size=4,
            vit_model_name=VIT_MODEL_NAME,
            pretrained=False 
        ).to(device)
        
        model.eval()
        
        # Compute GFLOPS
        flops = compute_gflops(model, device)
        results_gflops.append(flops)

    # --- PLOTTING (Strict ICML Style) ---
    # 1. Clear Plotting Style
    plt.style.use('seaborn-v0_8-white')
    
    # 2. Strict Parameter Overrides for Legibility & Reproduction
    mpl.rcParams.update({
        'font.family': 'serif',          # Serif font (e.g., Times) is standard
        'font.size': 14,                 # Legible text size
        'axes.labelsize': 16,            # Axis labels slightly larger
        'axes.linewidth': 1.5,           # Thicker axis spines
        'lines.linewidth': 2.5,          # Thicker lines (>0.5pt requirement)
        'lines.markersize': 8,           # Clear markers
        'xtick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.size': 6,
        'ytick.major.width': 1.5,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'figure.autolayout': True        # Prevent label clipping
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    
    # X-Axis Data: 100 -> 10
    patches_kept_pct = [p * 100 for p in patch_percentages]
    
    # Plot Line
    ax.plot(patches_kept_pct, results_gflops, marker='s', color='black', label='ViT-Large')
    
    # Labels (No Title inside figure!)
    ax.set_xlabel("Patches Kept (%)")
    ax.set_ylabel("GFLOPS")
    
    # Axes limits and formatting
    # X-Axis: Inverted (100% on Left -> 0% on Right) to show "compression"
    ax.set_xlim(105, 5)
    
    # Y-Axis: Start at 0 to avoid misleading scaling
    ax.set_ylim(bottom=0)
    
    # Add subtle grid for readability
    ax.grid(True, linestyle='--')
    
    # Save Output
    out_dir = project_root / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fig_gflops_scaling.pdf"
    
    # Save as PDF (Vector) and PNG (Preview)
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    
    print(f"\nSuccess! Plot saved to: {out_path}")

if __name__ == "__main__":
    main()