"""Compare all controllers on the correct datacenter model."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.datacenter_env import DatacenterCoolingEnv, load_weather
from agents.forecast_agent import ForecastDDPGAgent


def rule_based_controller(state, forecast):
    rad_capacity_normalized = state[4]
    rad_capacity_kW = rad_capacity_normalized * 500
    if rad_capacity_kW > 200:
        return 1.0
    elif rad_capacity_kW > 100:
        return 0.7
    elif rad_capacity_kW > 50:
        return 0.4
    else:
        return 0.1


def mpc_controller(state, forecast, env):
    rad_capacity_normalized = state[4]
    current_capacity = rad_capacity_normalized * 500
    future_temps = forecast[:, 0] * 40 + 10
    future_humidity = forecast[:, 1]
    future_clouds = forecast[:, 2]
    future_quality = (1 - future_temps/50) * (1 - future_humidity) * (1 - future_clouds)
    avg_future_quality = np.mean(future_quality)
    current_quality = (1 - state[0]) * (1 - state[1]) * (1 - state[2])
    
    if current_capacity > 300:
        return 1.0
    elif current_capacity > 100 and current_quality > avg_future_quality:
        return 0.9
    elif current_capacity > 100:
        return 0.6
    elif current_capacity > 50:
        return 0.4
    else:
        return 0.1


def evaluate_controller(env, controller_fn, name, num_episodes=10):
    elec_savings, water_savings, rewards = [], [], []
    
    for ep in range(num_episodes):
        start_hour = (ep * 876) % 8000
        state = env.reset(start_hour=start_hour)
        forecast = env.get_forecast()
        total_reward = 0
        
        for _ in range(env.episode_hours):
            action = controller_fn(state, forecast, env)
            state, reward, done, info = env.step(action)
            forecast = env.get_forecast()
            total_reward += reward
            if done:
                break
        
        elec_savings.append(info['electricity_savings_pct'])
        water_savings.append(info['water_savings_pct'])
        rewards.append(total_reward)
    
    return {'name': name, 'elec_mean': np.mean(elec_savings), 'elec_std': np.std(elec_savings),
            'water_mean': np.mean(water_savings), 'water_std': np.std(water_savings), 'reward_mean': np.mean(rewards)}


def main():
    print("=" * 70)
    print("CONTROLLER COMPARISON - CORRECT DATACENTER MODEL")
    print("=" * 70)
    
    weather_df = load_weather('data/weather/phoenix_az_tmy.csv')
    env = DatacenterCoolingEnv(weather_df, episode_hours=48)
    # Load TD3
    from train_comprehensive import ForecastTD3Agent, ForecastSACAgent
    
    td3_agent = ForecastTD3Agent(state_dim=7, forecast_horizon=6, forecast_features=3, action_dim=1)
    try:
        td3_agent.load('results/models/td3_best.pt')
        controllers['TD3'] = lambda s, f, e: td3_agent.select_action(s, f, add_noise=False)[0]
    except:
        print("Warning: Could not load TD3")
    
    sac_agent = ForecastSACAgent(state_dim=7, forecast_horizon=6, forecast_features=3, action_dim=1)
    try:
        sac_agent.load('results/models/sac_best.pt')
        controllers['SAC'] = lambda s, f, e: sac_agent.select_action(s, f, add_noise=False)[0]
    except:
        print("Warning: Could not load SAC")
        
    agent = ForecastDDPGAgent(state_dim=7, forecast_horizon=6, forecast_features=3, action_dim=1)
    try:
        agent.load('results/models/ddpg_correct.pt')
        ddpg_loaded = True
    except:
        print("Warning: Could not load DDPG agent")
        ddpg_loaded = False
    
    controllers = {
        'Tower Only': lambda s, f, e: 0.0,
        'Fixed 25%': lambda s, f, e: 0.25,
        'Fixed 50%': lambda s, f, e: 0.50,
        'Fixed 75%': lambda s, f, e: 0.75,
        'Fixed 100%': lambda s, f, e: 1.0,
        'Rule-Based': lambda s, f, e: rule_based_controller(s, f),
        'MPC': lambda s, f, e: mpc_controller(s, f, e),
    }
    
    if ddpg_loaded:
        controllers['DDPG (Ours)'] = lambda s, f, e: agent.select_action(s, f, add_noise=False)[0]
    
    print(f"\nEvaluating {len(controllers)} controllers...")
    print("-" * 70)
    
    results = []
    for name, ctrl_fn in controllers.items():
        result = evaluate_controller(env, ctrl_fn, name, num_episodes=10)
        results.append(result)
        print(f"{name:<15} | Elec: {result['elec_mean']:>6.1f}% ± {result['elec_std']:>4.1f}% | "
              f"Water: {result['water_mean']:>6.1f}% ± {result['water_std']:>4.1f}%")
    
    print("\n" + "=" * 70)
    print("SUMMARY - RANKED BY WATER SAVINGS")
    print("=" * 70)
    
    results_sorted = sorted(results, key=lambda x: x['water_mean'], reverse=True)
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['name']:<15} Electricity: {r['elec_mean']:>6.1f}% | Water: {r['water_mean']:>6.1f}%")


if __name__ == "__main__":
    main()
