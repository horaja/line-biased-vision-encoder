#!/usr/bin/env python3
"""
ICML-style Trade-off visualization script for SelectiveMagnoViT.
Recursively finds all evaluation_*.json files in the results/ directory.
"""

import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
import matplotlib as mpl

def setup_publication_style():
    """
    Configures Matplotlib to match ICML/Conference guidelines:
    - No gray backgrounds.
    - Legible fonts (approx 9pt+ equivalent).
    - Thick lines (>= 0.5pt).
    """
    # Use a clean style without gray grid backdrops
    plt.style.use('seaborn-v0_8-white')
    
    # Overrides for specific strict requirements
    mpl.rcParams.update({
        'font.family': 'serif',          # Serif often preferred for paper bodies (Times/Computer Modern)
        'font.size': 14,                 # Legible text
        'axes.labelsize': 16,            # Axis labels slightly larger
        'axes.linewidth': 1.5,           # Darker, thicker axes spines
        'lines.linewidth': 2.5,          # Thick lines for reproduction
        'lines.markersize': 8,           # Visible markers
        'xtick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.size': 6,
        'ytick.major.width': 1.5,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,               # Faint grid if used, or disable
        'figure.autolayout': True        # Ensure labels don't get cut off
    })

def main():
    setup_publication_style()

    # 1. SETUP PATHS
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    
    # Search recursively for json files
    files = list(results_dir.rglob("evaluation_*.json"))
    
    if not files:
        print(f"Error: No JSON files found in {results_dir} or subdirectories.")
        sys.exit(1)

    print(f"Found {len(files)} evaluation files. Parsing...")

    # 2. EXTRACT DATA
    data_points = []
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                run_data = json.load(f)
                
                if "patch_selection_stats" not in run_data:
                    continue
                
                stats = run_data["patch_selection_stats"]
                
                pct = stats.get("avg_patch_percentage", 0.0)
                acc = run_data.get("accuracy", 0.0)
                flops = run_data.get("gflops", 0.0)
                
                data_points.append({
                    "pct": pct, 
                    "acc": acc, 
                    "flops": flops
                })
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")

    if not data_points:
        print("No valid data points extracted.")
        sys.exit(1)

    # Sort data for clean line plotting (by patch percentage)
    data_points.sort(key=lambda x: x["pct"])

    # Convert to Lists & Percentages
    percentages = [x["pct"] * 100 for x in data_points]  # 0.5 -> 50.0
    accuracies = [x["acc"] * 100 for x in data_points]   # 0.75 -> 75.0
    flops_list = [x["flops"] for x in data_points]

    # Detect Baseline (Highest Patch %)
    baseline_run = max(data_points, key=lambda x: x["pct"])
    baseline_acc = baseline_run["acc"] * 100
    print(f"Baseline: {baseline_acc:.2f}% Accuracy")

    # ---------------------------------------------------------
    # PLOT 1: Accuracy vs. Patch Percentage
    # (X-Axis: 100% -> 0%)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4)) # Standard single-column paper figure size approx
    
    # Plot Data
    ax.plot(percentages, accuracies, marker='o', color='black', label='Ours')
    
    # Plot Baseline Reference
    ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Baseline')

    # Formatting
    ax.set_xlabel("Patches Kept (%)")
    ax.set_ylabel("Accuracy (%)")
    
    # Fix Misleading Scales: Set Y to 0-100 or a generous range
    # If your accuracy is high (e.g. 80-90), ylim(0, 100) is honest but hard to see differences.
    # ICML often accepts "zoomed" if clearly labeled, but "honest" starts at 0.
    # Adjust this based on your preference. Defaulting to 0-100 for absolute honesty.
    ax.set_ylim(0, 105) 
    
    # Invert X Axis: Larger (100%) -> Smaller (0%)
    ax.set_xlim(105, -5) # Small padding
    
    # Clean up
    ax.grid(True, linestyle='--')
    ax.legend(frameon=True, fancybox=False, edgecolor='black') # Simple legend
    
    # Save
    out_path = results_dir / "fig_accuracy_vs_patches.pdf" # PDF is better for LaTeX
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches='tight') # Backup PNG
    print(f"Saved: {out_path}")

    # ---------------------------------------------------------
    # PLOT 2: Accuracy vs. GFLOPS
    # (X-Axis: High GFLOPS -> Low GFLOPS)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot Data
    ax.plot(flops_list, accuracies, marker='s', color='black', label='Ours')
    
    # Baseline
    ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Baseline')

    # Formatting
    ax.set_xlabel("GFLOPS") # Removed "Lower is better" to keep it clean, direction implies it
    ax.set_ylabel("Accuracy (%)")
    
    ax.set_ylim(0, 105)
    
    # Invert X Axis: High Compute -> Low Compute
    max_flops = max(flops_list) if flops_list else 1
    ax.set_xlim(max_flops * 1.1, -0.1) 
    
    # Clean up
    ax.grid(True, linestyle='--')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    
    # Save
    out_path = results_dir / "fig_accuracy_vs_gflops.pdf"
    plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()