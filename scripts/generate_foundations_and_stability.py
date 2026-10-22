import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, glob

# Set Academic Style
plt.style.use('seaborn-v0_8-paper')

def generate_suite():
    if not os.path.exists('results'): os.makedirs('results')

    # --- FIG 2: CHILLER COP VALIDATION ---
    temp = np.linspace(10, 45, 50)
    cop = 6.5 * (1 - 0.008 * (temp - 20)) 
    plt.figure(figsize=(8, 5))
    plt.plot(temp, cop, color='red', linewidth=2, label='Chiller Model')
    plt.title('FIG 2: Chiller Efficiency (COP) vs. Ambient Temperature')
    plt.xlabel('Outdoor Temperature (°C)')
    plt.ylabel('Coefficient of Performance (COP)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/fig2_cop_vs_temperature.png', dpi=300)
    print("Generated FIG 2")

    # --- FIG 3: RADIATIVE CAPACITY ---
    hours = np.arange(24)
    capacity = 70 + 30 * np.cos(2 * np.pi * (hours - 2) / 24) 
    plt.figure(figsize=(8, 5))
    plt.fill_between(hours, capacity, color='skyblue', alpha=0.4)
    plt.plot(hours, capacity, color='blue', linewidth=2)
    plt.title('FIG 3: Hourly Radiative Cooling Potential (5,000 m² Array)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Cooling Power Density (W/m²)')
    plt.savefig('results/fig3_radiative_capacity_daily.png', dpi=300)
    print("Generated FIG 3")

    # --- FIG 6: TRAINING CURVES (Learning Stability) ---
    log_files = glob.glob("results/logs/*.csv")
    if log_files:
        plt.figure(figsize=(10, 6))
        for f in log_files:
            name = os.path.basename(f).replace(".csv", "")
            df = pd.read_csv(f)
            # Smooth the rewards for a professional look
            plt.plot(df['episode'], df['reward'].rolling(window=30).mean(), label=name, alpha=0.7)
        plt.title('FIG 6: Deep RL Training Convergence Stability')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Episodic Reward')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small', ncol=2)
        plt.savefig('results/fig6_training_curve.png', dpi=300, bbox_inches='tight')
        print("Generated FIG 6")
    else:
        print("SKIP FIG 6: No log files found in results/logs/")

    # FIG 9 removed (Oct 2026): it was a bar chart of two hardcoded numbers (95 vs 102 W/m2)
    # labelled as a validation against SkyCool. No measured data or model output was involved.

if __name__ == "__main__":
    generate_suite()
