import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Silent Parsing Logic ---
def parse_evaluation_results(root_dir="results"):
    data = []
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found.")
        return pd.DataFrame()

    for current_folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(current_folder, file)
                
                # Path parsing
                experiment_id = os.path.basename(current_folder)
                parent_folder = os.path.dirname(current_folder)
                method = os.path.basename(parent_folder) # 'smart' or 'random'

                if not experiment_id.startswith("eval_"):
                    continue

                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        
                        # Extract metrics (handling nesting)
                        accuracy = content.get('accuracy', 0)
                        
                        patch_stats = content.get('patch_selection_stats', {})
                        keep_ratio = patch_stats.get('avg_patch_percentage', None)
                        
                        if keep_ratio is None:
                            keep_ratio = content.get('keep_ratio', 0)

                        data.append({
                            "method": method,
                            "experiment_id": experiment_id,
                            "accuracy": accuracy,
                            "keep_ratio": keep_ratio
                        })
                        
                except Exception:
                    pass # Silent failure for bad files

    return pd.DataFrame(data)

# --- 2. ICML Style Plotting ---
def plot_icml_style(df, output_dir="results"):
    # Set style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (6, 4),
        'lines.linewidth': 2.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':'
    })

    plt.figure()

    # Data conversion
    df['Patches Kept (%)'] = df['keep_ratio'] * 100
    df['Accuracy (%)'] = df['accuracy'] * 100

    # Define styles
    # Mapping: smart -> Ours, random -> Random
    styles = {
        'smart':  {'color': 'black', 'ls': '-',  'marker': 'o', 'label': 'Ours'},
        'random': {'color': 'grey',  'ls': '--', 'marker': '',  'label': 'Random'}
    }

    # Plot
    for method in ['smart', 'random']:
        subset = df[df['method'] == method]
        
        # Sort so lines connect properly
        subset = subset.sort_values(by='Patches Kept (%)', ascending=False)
        
        if subset.empty:
            continue
            
        style = styles.get(method, {'color': 'blue', 'ls': '-', 'marker': 'x', 'label': method})

        plt.plot(
            subset['Patches Kept (%)'], 
            subset['Accuracy (%)'], 
            label=style['label'],
            color=style['color'],
            linestyle=style['ls'],
            marker=style['marker'],
            markersize=8,
            linewidth=2.5
        )

    # Formatting
    plt.xlabel('Patches Kept (%)')
    plt.ylabel('Accuracy (%)')
    
    # 1. Start Scale at 0
    plt.ylim(0, 100) 
    
    # 2. Invert X axis (100 -> 0)
    plt.gca().invert_xaxis()
    
    # Legend
    plt.legend(frameon=True, fancybox=False, edgecolor='black')
    
    # Clean spines
    sns.despine()

    # 3. Save to results folder
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'icml_figure.png')
    
    # Ensure directory exists just in case
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Done. Plot saved to {output_path}")

# --- Execution ---
if __name__ == "__main__":
    target_dir = "results"
    df = parse_evaluation_results(target_dir)
    
    if not df.empty:
        plot_icml_style(df, output_dir=target_dir)
    else:
        print("No valid data found.")