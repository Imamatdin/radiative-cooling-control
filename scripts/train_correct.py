"""Train DDPG on Correct Datacenter Cooling Environment."""

import numpy as np
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.datacenter_env import DatacenterCoolingEnv, load_weather
from agents.forecast_agent import ForecastDDPGAgent


def train():
    print("=" * 70)
    print("DDPG TRAINING - CORRECT DATACENTER MODEL")
    print("=" * 70)
    
    weather_df = load_weather('data/weather/phoenix_az_tmy.csv')
    print(f"Weather: {len(weather_df)} hours (Phoenix, AZ)")
    
    env = DatacenterCoolingEnv(weather_df, episode_hours=48)
    agent = ForecastDDPGAgent(state_dim=7, forecast_horizon=6, forecast_features=3, action_dim=1,
                              hidden_dims=[256, 256], actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005, batch_size=128)
    
    # Warmup
    print("\nWarmup: filling replay buffer...")
    state = env.reset()
    for _ in range(3000):
        action = np.random.rand()
        forecast = env.get_forecast()
        next_state, reward, done, _ = env.step(action)
        next_forecast = env.get_forecast()
        agent.store_transition(state, forecast, np.array([action]), reward, next_state, next_forecast, done)
        if done:
            state = env.reset()
        else:
            state = next_state
    print(f"  Buffer: {len(agent.buffer)} transitions")
    
    # Training
    num_episodes = 500
    best_reward = -float('inf')
    rewards_history = []
    start_time = time.time()
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        forecast = env.get_forecast()
        ep_reward = 0
        
        for step in range(env.episode_hours):
            action = agent.select_action(state, forecast, add_noise=True)
            next_state, reward, done, info = env.step(action[0])
            next_forecast = env.get_forecast()
            agent.store_transition(state, forecast, action, reward, next_state, next_forecast, done)
            for _ in range(2):
                agent.train_step()
            ep_reward += reward
            state, forecast = next_state, next_forecast
            if done:
                break
        
        rewards_history.append(ep_reward)
        
        if ep % 10 == 0:
            print(f"Ep {ep:3d}/{num_episodes} | Reward: {ep_reward:7.1f} | "
                  f"Elec: {info['electricity_savings_pct']:5.1f}% | Water: {info['water_savings_pct']:5.1f}%")
        
        if ep % 50 == 0:
            eval_rewards = []
            for _ in range(5):
                s = env.reset()
                f = env.get_forecast()
                er = 0
                for _ in range(env.episode_hours):
                    a = agent.select_action(s, f, add_noise=False)
                    s, r, d, info = env.step(a[0])
                    f = env.get_forecast()
                    er += r
                    if d:
                        break
                eval_rewards.append(er)
            avg_reward = np.mean(eval_rewards)
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save('results/models/ddpg_correct.pt')
                print(f"  *** Best model saved! Reward={avg_reward:.1f} ***")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes. Best reward: {best_reward:.1f}")
    
    results = {'best_reward': best_reward, 'rewards_history': [float(r) for r in rewards_history]}
    with open('results/ddpg_correct_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return agent, results


if __name__ == "__main__":
    agent, results = train()
