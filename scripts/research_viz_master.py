import json, os, glob
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Force headless mode for RunPod
import matplotlib.pyplot as plt
import seaborn as sns

# Professional Style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def generate_master_suite():
    json_path = 'results/multicity_results.json'
    
    if not os.path.exists(json_path):
        print(f"CRITICAL ERROR: {json_path} not found. Are you in the root directory?")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Prepare Main Data
    records = []
    cities = ['phoenix', 'houston', 'seattle']
    for city in cities:
        for algo in ['DDPG', 'TD3', 'SAC', 'Fixed 100%', 'Fixed 50%', 'Tower Only']:
            key = f"{algo}_{city}"
            if key in data.get('evaluation', {}):
                res = data['evaluation'][key]['test']
                records.append({
                    'City': city.capitalize(), 'Algorithm': algo,
                    'Water Savings (%)': res['water_mean'], 'Elec Savings (%)': res['elec_mean']
                })
    
    if not records:
        print("ERROR: No evaluation data found in JSON. Check your keys.")
        return

    df = pd.DataFrame(records)
    print(f"Successfully processed {len(records)} results.")

    # FIG 4: Head-to-Head
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='City', y='Water Savings (%)', hue='Algorithm', palette='viridis')
    plt.title('FIG 4: Multi-Agent Water Efficiency (1 MW Scale)')
    plt.savefig('results/fig4_controller_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved FIG 4.")

    # FIG 5: Scatter Performance
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Elec Savings (%)', y='Water Savings (%)', hue='Algorithm', style='City', s=200)
    plt.title('FIG 5: Efficiency Frontier (1 MW)')
    plt.savefig('results/fig5_annual_performance.png', dpi=300, bbox_inches='tight')
    print("Saved FIG 5.")

    # FIG 8: Ablation (Forecast vs No Forecast)
    abl_records = []
    for city in cities:
        for variant in ['forecast', 'no_forecast']:
            key = f"DDPG_{variant}_{city}"
            if key in data.get('ablation', {}):
                res = data['ablation'][key]['test']
                abl_records.append({
                    'City': city.capitalize(),
                    'Predictive': 'With Forecast' if 'no' not in variant else 'No Forecast',
                    'Water Savings (%)': res['water_mean']
                })
    
    if abl_records:
        df_abl = pd.DataFrame(abl_records)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_abl, x='City', y='Water Savings (%)', hue='Predictive', palette='magma')
        plt.title('FIG 8: Value of Predictive Intelligence')
        plt.savefig('results/fig8_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved FIG 8.")

    # FIG 6: Training Curves
    log_files = glob.glob("results/logs/*.csv")
    if log_files:
        plt.figure(figsize=(12, 6))
        for f in log_files:
            name = os.path.basename(f).replace(".csv", "")
            ldf = pd.read_csv(f)
            plt.plot(ldf['episode'], ldf['reward'].rolling(window=30).mean(), label=name, alpha=0.6)
        plt.title('FIG 6: Deep RL Training Convergence')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small', ncol=2)
        plt.savefig('results/fig6_training_curve.png', dpi=300, bbox_inches='tight')
        print("Saved FIG 6.")
    else:
        print("SKIP: No .csv files found in results/logs/ for FIG 6.")

if __name__ == "__main__":
    generate_master_suite()
