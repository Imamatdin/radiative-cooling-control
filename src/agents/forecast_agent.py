"""
Forecast-Aware DDPG Agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class ForecastActor(nn.Module):
    def __init__(self, state_dim: int = 5, forecast_dim: int = 18, action_dim: int = 1, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        input_dim = state_dim + forecast_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, action_dim)
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, forecast: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, forecast], dim=-1)
        x = self.features(x)
        return torch.tanh(self.output(x))


class ForecastCritic(nn.Module):
    def __init__(self, state_dim: int = 5, forecast_dim: int = 18, action_dim: int = 1, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        input_dim = state_dim + forecast_dim + action_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
    
    def forward(self, state: torch.Tensor, forecast: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, forecast, action], dim=-1)
        x = self.features(x)
        return self.output(x)


class ForecastReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, forecast, action, reward, next_state, next_forecast, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, forecast, action, reward, next_state, next_forecast, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([t[0] for t in batch])
        forecasts = np.array([t[1] for t in batch])
        actions = np.array([t[2] for t in batch])
        rewards = np.array([t[3] for t in batch])
        next_states = np.array([t[4] for t in batch])
        next_forecasts = np.array([t[5] for t in batch])
        dones = np.array([t[6] for t in batch])
        return states, forecasts, actions, rewards, next_states, next_forecasts, dones
    
    def __len__(self):
        return len(self.buffer)


class ForecastDDPGAgent:
    def __init__(self, state_dim: int = 5, forecast_horizon: int = 6, forecast_features: int = 3,
                 action_dim: int = 1, hidden_dims: List[int] = [256, 256], actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 buffer_capacity: int = 100000, batch_size: int = 128, device: str = None):
        self.state_dim = state_dim
        self.forecast_dim = forecast_horizon * forecast_features
        self.forecast_horizon = forecast_horizon
        self.forecast_features = forecast_features
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.actor = ForecastActor(state_dim, self.forecast_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = ForecastActor(state_dim, self.forecast_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = ForecastCritic(state_dim, self.forecast_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = ForecastCritic(state_dim, self.forecast_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = ForecastReplayBuffer(buffer_capacity)
        
        self.noise_scale = 0.2
        self.noise_decay = 0.9999
        self.noise_min = 0.01
    
    def select_action(self, state: np.ndarray, forecast: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        forecast_t = torch.FloatTensor(forecast.flatten()).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t, forecast_t).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = action + noise
            self.noise_scale = max(self.noise_min, self.noise_scale * self.noise_decay)
        
        action = np.clip(action, -1, 1)
        action = (action + 1) / 2
        return action
    
    def store_transition(self, state, forecast, action, reward, next_state, next_forecast, done):
        action_stored = action * 2 - 1
        self.buffer.push(state, forecast.flatten(), action_stored, reward, next_state, next_forecast.flatten(), done)
    
    def train_step(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}
        
        states, forecasts, actions, rewards, next_states, next_forecasts, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        forecasts = torch.FloatTensor(forecasts).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_forecasts = torch.FloatTensor(next_forecasts).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states, next_forecasts)
            target_q = self.critic_target(next_states, next_forecasts, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, forecasts, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        predicted_actions = self.actor(states, forecasts)
        actor_loss = -self.critic(states, forecasts, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'noise_scale': self.noise_scale}
    
    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_scale': self.noise_scale,
        }, path)
        print(f"Forecast DDPG agent saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.noise_scale = checkpoint.get('noise_scale', 0.1)
        print(f"Forecast DDPG agent loaded from {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("FORECAST-AWARE DDPG - VALIDATION")
    print("=" * 60)
    
    agent = ForecastDDPGAgent(state_dim=5, forecast_horizon=6, forecast_features=3, action_dim=1)
    print(f"Agent configuration: State dim: {agent.state_dim}, Forecast dim: {agent.forecast_dim}")
    
    state = np.random.rand(5)
    forecast = np.random.rand(6, 3)
    action = agent.select_action(state, forecast, add_noise=False)
    print(f"Test action: {action}")
    
    for i in range(200):
        s = np.random.rand(5)
        f = np.random.rand(6, 3)
        a = agent.select_action(s, f, add_noise=True)
        r = np.random.randn()
        ns = np.random.rand(5)
        nf = np.random.rand(6, 3)
        agent.store_transition(s, f, a, r, ns, nf, False)
    
    for i in range(5):
        metrics = agent.train_step()
        print(f"Step {i+1}: {metrics}")
    
    print("FORECAST DDPG VALIDATED")
