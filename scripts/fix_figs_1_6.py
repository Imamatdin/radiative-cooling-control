import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os, glob

def generate_fig1_schematic():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    
    # Draw Components
    ax.add_patch(patches.Rectangle((1, 2), 2, 2, color='grey', alpha=0.3)) # Data Center
    ax.text(2, 3, "1 MW Pod", ha='center', fontweight='bold')
    
    ax.add_patch(patches.Circle((5, 3), 0.8, color='blue', alpha=0.3)) # Tank
    ax.text(5, 3, "50m³ Tank", ha='center', fontweight='bold')
    
    ax.add_patch(patches.Rectangle((7, 4), 2, 1, color='skyblue', alpha=0.3)) # Panels
    ax.text(8, 4.5, "5,000m² Panels", ha='center', fontweight='bold')
    
    # Draw Connections
    ax.annotate("", xy=(4.2, 3), xytext=(3, 3), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(7, 4.5), xytext=(5.8, 3.5), arrowprops=dict(arrowstyle="->", lw=2))
    
    plt.title("FIG 1: System Architecture Overview")
    plt.savefig('results/fig1_system_architecture.png', dpi=300)
    print("Forced FIG 1 generation.")

def generate_fig6_training():
    # Look for logs in common locations
    log_dirs = ['results/logs', 'results', './']
    found_files = []
    for d in log_dirs:
        found_files.extend(glob.glob(os.path.join(d, "*.csv")))
    
    if found_files:
        plt.figure(figsize=(10, 6))
        for f in found_files[:5]: # Limit to 5 for clarity
            name = os.path.basename(f)
            df = pd.read_csv(f)
            if 'reward' in df.columns:
                plt.plot(df['episode'], df['reward'].rolling(window=30).mean(), label=name)
        plt.title("FIG 6: Training Convergence (Learning Curves)")
        plt.legend(fontsize='small')
        plt.savefig('results/fig6_training_curve.png', dpi=300)
        print(f"Generated FIG 6 using: {[os.path.basename(x) for x in found_files[:5]]}")
    else:
        print("CRITICAL: Still no .csv logs found. Run 'ls -R' to find where your logs are saved.")

if __name__ == "__main__":
    generate_fig1_schematic()
    generate_fig6_training()
