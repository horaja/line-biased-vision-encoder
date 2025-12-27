import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# --- 1. Parsing Logic ---
def parse_evaluation_results(root_dir="results"):
    data = []
    
    # Only process these specific folders
    valid_methods = {'smart', 'random', 'few', 'few-random'}
    
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
                method = os.path.basename(parent_folder) 

                # Filter: Must be in the valid list and must be an eval folder
                if method not in valid_methods:
                    continue
                if not experiment_id.startswith("eval_"):
                    continue

                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        
                        # Extract metrics
                        accuracy = content.get('accuracy', 0)
                        
                        patch_stats = content.get('patch_selection_stats', {})
                        keep_ratio = patch_stats.get('avg_patch_percentage', None)
                        
                        # Fallback for older/different json structures
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

# --- 2. Combined Plotting with Log Scale ---
def plot_combined_tradeoffs(df, output_dir="results"):
    # Set ICML/Paper style
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

    # Data conversion to percentages
    df['Patches Kept (%)'] = df['keep_ratio'] * 100
    df['Accuracy (%)'] = df['accuracy'] * 100

    # Create a Group column to merge curves
    # smart + few -> Ours
    # random + few-random -> Random
    method_map = {
        'smart': 'Ours',
        'few': 'Ours',
        'random': 'Random',
        'few-random': 'Random'
    }
    df['group'] = df['method'].map(method_map)

    # Define styles for the two main groups
    styles = {
        'Ours':   {'color': 'black', 'ls': '-',  'marker': 'o', 'label': 'Ours'},
        'Random': {'color': 'grey',  'ls': '--', 'marker': '',  'label': 'Random'}
    }

    # Plot each group
    for group_name in ['Ours', 'Random']:
        subset = df[df['group'] == group_name]
        
        # Sort by patch percentage descending (100 -> 0)
        subset = subset.sort_values(by='Patches Kept (%)', ascending=False)
        
        if subset.empty:
            print(f"Warning: No data found for group '{group_name}'")
            continue
            
        style = styles.get(group_name)

        plt.plot(
            subset['Patches Kept (%)'], 
            subset['Accuracy (%)'], 
            label=style['label'],
            color=style['color'],
            linestyle=style['ls'],
            marker=style['marker'],
            markersize=6,
            linewidth=2.5
        )

    # Formatting
    plt.xlabel('Patches Kept (%) (Log Scale)')
    plt.ylabel('Accuracy (%)')
    
    # 1. Log Scale for X Axis to un-bunch the small values
    plt.xscale('log')
    
    # 2. Set nice integer ticks for log scale (1, 2, 5, 10, 20, 50, 100)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().set_xticks([1, 2, 5, 10, 20, 50, 100])
    
    # 3. Start scale at 0 for Y, and fit X tightly
    plt.ylim(0, 100)
    plt.xlim(100, 0.9) # Limits slightly wider than 1 to ensure 1 is visible
    
    # 4. Invert X axis (100 -> 1)
    # Note: For log scale, we set limits (high, low) to invert
    plt.xlim(105, 0.9) 
    
    # Legend
    plt.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right')
    
    # Clean spines
    sns.despine()

    # Save
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'combined_tradeoff_log_figure.png')
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Done. Plot saved to {output_path}")

# --- Execution ---
if __name__ == "__main__":
    target_dir = "results"
    
    # 1. Parse
    print(f"Scanning '{target_dir}' for smart, random, few, few-random...")
    df = parse_evaluation_results(target_dir)
    
    # 2. Plot
    if not df.empty:
        print("Found data for methods:", df['method'].unique())
        plot_combined_tradeoffs(df, output_dir=target_dir)
    else:
        print("No valid data found matching the criteria.")