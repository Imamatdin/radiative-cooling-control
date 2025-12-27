"""
Evaluate trained RL agents on the datacenter cooling environment.

Loads saved models and evaluates performance across different scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.datacenter_env import DatacenterCoolingEnv, load_weather
from agents.forecast_agent import ForecastDDPGAgent


class Evaluator:
    """Evaluate trained agents on datacenter cooling."""
    
    def __init__(self, weather_df: pd.DataFrame, agent_path: str = None):
        self.weather_df = weather_df
        self.env = DatacenterCoolingEnv(weather_df, episode_hours=168)  # 1 week
        
        self.agent = None
        if agent_path and os.path.exists(agent_path):
            self.agent = ForecastDDPGAgent(
                state_dim=7, forecast_horizon=6, forecast_features=3, action_dim=1
            )
            self.agent.load(agent_path)
            print(f"Loaded agent from {agent_path}")
    
    def evaluate_episode(self, controller_fn, start_hour: int = None) -> dict:
        """Run single evaluation episode."""
        state = self.env.reset(start_hour=start_hour)
        forecast = self.env.get_forecast()
        
        total_reward = 0
        actions = []
        infos = []
        
        for step in range(self.env.episode_hours):
            action = controller_fn(state, forecast)
            next_state, reward, done, info = self.env.step(action)
            forecast = self.env.get_forecast()
            
            total_reward += reward
            actions.append(action)
            infos.append(info)
            
            state = next_state
            if done:
                break
        
        return {
            'total_reward': total_reward,
            'electricity_savings_pct': info['electricity_savings_pct'],
            'water_savings_pct': info['water_savings_pct'],
            'actions': actions,
            'infos': infos,
        }
    
    def evaluate_multi_episode(self, controller_fn, controller_name: str, 
                               num_episodes: int = 20) -> dict:
        """Evaluate controller across multiple episodes."""
        results = {
            'rewards': [],
            'elec_savings': [],
            'water_savings': [],
        }
        
        for ep in range(num_episodes):
            # Sample different starting points throughout the year
            start_hour = (ep * 400) % (len(self.weather_df) - 200)
            episode_result = self.evaluate_episode(controller_fn, start_hour)
            
            results['rewards'].append(episode_result['total_reward'])
            results['elec_savings'].append(episode_result['electricity_savings_pct'])
            results['water_savings'].append(episode_result['water_savings_pct'])
        
        summary = {
            'name': controller_name,
            'reward_mean': np.mean(results['rewards']),
            'reward_std': np.std(results['rewards']),
            'elec_mean': np.mean(results['elec_savings']),
            'elec_std': np.std(results['elec_savings']),
            'water_mean': np.mean(results['water_savings']),
            'water_std': np.std(results['water_savings']),
        }
        
        return summary
    
    def run_full_evaluation(self) -> pd.DataFrame:
        """Evaluate all controllers."""
        controllers = {
            'Tower Only': lambda s, f: 0.0,
            'Fixed 50%': lambda s, f: 0.5,
            'Fixed 100%': lambda s, f: 1.0,
            'Rule-Based': lambda s, f: min(1.0, s[4] / 0.4),  # Based on rad capacity
            'Night Priority': lambda s, f: 1.0 if s[6] > 0.5 else 0.3,
        }
        
        if self.agent is not None:
            controllers['DDPG (Ours)'] = lambda s, f: self.agent.select_action(s, f, add_noise=False)[0]
        
        results = []
        for name, ctrl in controllers.items():
            print(f"Evaluating {name}...")
            summary = self.evaluate_multi_episode(ctrl, name, num_episodes=20)
            results.append(summary)
            print(f"  Elec: {summary['elec_mean']:.1f}% ± {summary['elec_std']:.1f}%")
            print(f"  Water: {summary['water_mean']:.1f}% ± {summary['water_std']:.1f}%")
        
        return pd.DataFrame(results)
    
    def generate_evaluation_figures(self, results_df: pd.DataFrame):
        """Generate evaluation visualizations."""
        os.makedirs('figures', exist_ok=True)
        
        # Bar chart comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(results_df))
        width = 0.6
        
        # Electricity savings
        colors = ['green' if v > 0 else 'red' for v in results_df['elec_mean']]
        bars1 = axes[0].bar(x, results_df['elec_mean'], width, 
                           yerr=results_df['elec_std'], capsize=5, color=colors, alpha=0.7)
        axes[0].set_ylabel('Electricity Savings (%)')
        axes[0].set_title('Electricity Savings by Controller')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df['name'], rotation=45, ha='right')
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Water savings
        colors = ['teal' if v > 0 else 'orange' for v in results_df['water_mean']]
        bars2 = axes[1].bar(x, results_df['water_mean'], width,
                           yerr=results_df['water_std'], capsize=5, color=colors, alpha=0.7)
        axes[1].set_ylabel('Water Savings (%)')
        axes[1].set_title('Water Savings by Controller')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results_df['name'], rotation=45, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('figures/evaluation_comparison.png', dpi=150)
        plt.close()
        print("✓ Saved figures/evaluation_comparison.png")


def main():
    print("=" * 70)
    print("AGENT EVALUATION - DATACENTER RADIATIVE COOLING")
    print("=" * 70)
    
    weather_df = load_weather('data/weather/phoenix_az_tmy.csv')
    print(f"Loaded {len(weather_df)} hours of weather data")
    
    evaluator = Evaluator(
        weather_df, 
        agent_path='results/models/ddpg_correct.pt'
    )
    
    results_df = evaluator.run_full_evaluation()
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    evaluator.generate_evaluation_figures(results_df)
    
    # Save results
    results_df.to_csv('results/evaluation_results.csv', index=False)
    print("\n✓ Results saved to results/evaluation_results.csv")


if __name__ == "__main__":
    main()
