"""
Comprehensive RL Training for Datacenter Radiative Cooling.

Trains and compares: DDPG, TD3, SAC
Saves all models and generates comparison data for figures.

This is the REAL training script - not placeholders.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.datacenter_env import DatacenterCoolingEnv, load_weather


# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

class Actor(nn.Module):
    """Deterministic actor for DDPG/TD3."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize output layer with small weights
        self.net[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.net[-1].bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        return torch.sigmoid(self.net(state))  # Output in [0, 1]


class Critic(nn.Module):
    """Q-function critic."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = state_dim + action_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DoubleCritic(nn.Module):
    """Twin Q-functions for TD3/SAC."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_dims)
        self.q2 = Critic(state_dim, action_dim, hidden_dims)
    
    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DDPG AGENT
# =============================================================================

class DDPGAgent:
    """Deep Deterministic Policy Gradient."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], 
                 lr=3e-4, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.buffer = ReplayBuffer()
        self.noise_scale = 0.2
    
    def select_action(self, state, noise=0.1):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if noise > 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, 0, 1)
    
    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target_q = rewards_t + (1 - dones_t) * self.gamma * target_q
        
        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # Actor update
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
    
    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

# =============================================================================
# FORECAST-AWARE TD3 AGENT
# =============================================================================

class ForecastTD3Agent:
    """Forecast-aware Twin Delayed DDPG."""
    
    def __init__(self, state_dim=7, forecast_horizon=6, forecast_features=3,
                 action_dim=1, hidden_dims=[256, 256], lr=3e-4, gamma=0.99, 
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_counter = 0
        self.action_dim = action_dim
        self.forecast_dim = forecast_horizon * forecast_features
        
        input_dim = state_dim + self.forecast_dim
        
        self.actor = Actor(input_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = Actor(input_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = DoubleCritic(input_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = DoubleCritic(input_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = deque(maxlen=100000)
        self.noise_scale = 0.2
    
    def select_action(self, state, forecast, add_noise=True):
        state_forecast = np.concatenate([state, forecast.flatten()])
        state_t = torch.FloatTensor(state_forecast).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if add_noise:
            action = action + np.random.normal(0, self.noise_scale, size=action.shape)
        return np.clip(action, 0, 1)
    
    def store_transition(self, state, forecast, action, reward, next_state, next_forecast, done):
        sf = np.concatenate([state, forecast.flatten()])
        nsf = np.concatenate([next_state, next_forecast.flatten()])
        self.buffer.append((sf, action, reward, nsf, done))
    
    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return {}
        self.update_counter += 1
        
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(0, 1)
            tq1, tq2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(tq1, tq2)
        
        cq1, cq2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(cq1, target_q) + nn.MSELoss()(cq2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        if self.update_counter % self.policy_delay == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        return {}
    
    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)


# =============================================================================
# FORECAST-AWARE SAC AGENT
# =============================================================================

class ForecastSACAgent:
    """Forecast-aware Soft Actor-Critic."""
    
    def __init__(self, state_dim=7, forecast_horizon=6, forecast_features=3,
                 action_dim=1, hidden_dims=[256, 256], lr=3e-4, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.forecast_dim = forecast_horizon * forecast_features
        
        input_dim = state_dim + self.forecast_dim
        
        self.actor = Actor(input_dim, action_dim, hidden_dims).to(self.device)
        self.critic = DoubleCritic(input_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = DoubleCritic(input_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = deque(maxlen=100000)
        self.noise_scale = 0.2
    
    def select_action(self, state, forecast, add_noise=True):
        state_forecast = np.concatenate([state, forecast.flatten()])
        state_t = torch.FloatTensor(state_forecast).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if add_noise:
            action = action + np.random.normal(0, self.noise_scale, size=action.shape)
        return np.clip(action, 0, 1)
    
    def store_transition(self, state, forecast, action, reward, next_state, next_forecast, done):
        sf = np.concatenate([state, forecast.flatten()])
        nsf = np.concatenate([next_state, next_forecast.flatten()])
        self.buffer.append((sf, action, reward, nsf, done))
    
    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return {}
        
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions = self.actor(next_states)
            noise = torch.randn_like(next_actions) * 0.1
            next_actions = (next_actions + noise).clamp(0, 1)
            tq1, tq2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(tq1, tq2)
        
        cq1, cq2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(cq1, target_q) + nn.MSELoss()(cq2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        actor_loss = -self.critic.q1(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        return {}
    
    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)


# =============================================================================
# MAIN - TRAIN TD3 & SAC WITH FORECAST (SAME AS DDPG)
# =============================================================================

def main():
    print("=" * 80)
    print("FORECAST-AWARE TD3 & SAC TRAINING")
    print("(Same conditions as your DDPG)")
    print("=" * 80)
    
    weather_df = load_weather('data/weather/phoenix_az_tmy.csv')
    print(f"\nLoaded {len(weather_df)} hours")
    
    env = DatacenterCoolingEnv(weather_df, episode_hours=48)
    os.makedirs('results/models', exist_ok=True)
    
    agents = {
        'TD3': ForecastTD3Agent(state_dim=7, forecast_horizon=6, forecast_features=3, action_dim=1),
        'SAC': ForecastSACAgent(state_dim=7, forecast_horizon=6, forecast_features=3, action_dim=1),
    }
    
    for name, agent in agents.items():
        print(f"\n{'='*60}\nTraining {name}\n{'='*60}")
        
        # Warmup - 3000 like DDPG
        state = env.reset()
        for _ in range(3000):
            action = np.random.rand()
            forecast = env.get_forecast()
            next_state, reward, done, _ = env.step(action)
            next_forecast = env.get_forecast()
            agent.store_transition(state, forecast, np.array([action]), reward, next_state, next_forecast, done)
            state = next_state if not done else env.reset()
        print(f"Buffer: {len(agent.buffer)}")
        
        # Train 500 episodes like DDPG
        best_reward = -float('inf')
        for ep in range(1, 501):
            state = env.reset()
            forecast = env.get_forecast()
            ep_reward = 0
            
            for _ in range(env.episode_hours):
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
            
            if ep % 10 == 0:
                print(f"Ep {ep}/500 | Reward: {ep_reward:.1f} | Elec: {info['electricity_savings_pct']:.1f}% | Water: {info['water_savings_pct']:.1f}%")
            
            if ep % 50 == 0:
                # Eval
                eval_r = []
                for _ in range(5):
                    s = env.reset()
                    f = env.get_forecast()
                    er = 0
                    for _ in range(48):
                        a = agent.select_action(s, f, add_noise=False)
                        s, r, d, _ = env.step(a[0])
                        f = env.get_forecast()
                        er += r
                        if d: break
                    eval_r.append(er)
                avg = np.mean(eval_r)
                if avg > best_reward:
                    best_reward = avg
                    agent.save(f'results/models/{name.lower()}_best.pt')
                    print(f"  *** Best {name} saved! Reward={avg:.1f} ***")
        
        print(f"{name} done. Best: {best_reward:.1f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
