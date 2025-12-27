"""
DDPG Agent and Networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional
import os


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple = (256, 256),
                 action_low: float = 0.0, action_high: float = 1.0):
        super(Actor, self).__init__()
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hidden:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        self.output.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.hidden(state)
        x = self.output(x)
        action = torch.tanh(x) * self.action_scale + self.action_bias
        return action

    def get_action(self, state: np.ndarray, device: str = 'cpu') -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.forward(state_tensor)
        return action.cpu().numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple = (256, 256)):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])
        self.output = nn.Linear(hidden_dims[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        self.output.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.output(x)


class DDPGNetworks:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple = (256, 256),
                 action_low: float = 0.0, action_high: float = 1.0, device: str = 'cpu'):
        self.device = device
        self.actor = Actor(state_dim, action_dim, hidden_dims, action_low, action_high).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims, action_low, action_high).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.hard_update()

    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update(self, tau: float = 0.005):
        for target_param, main_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
        for target_param, main_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def save(self, filepath: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (256, 256),
                 action_low: float = 0.0, action_high: float = 1.0, gamma: float = 0.99,
                 tau: float = 0.005, actor_lr: float = 1e-4, critic_lr: float = 1e-3,
                 buffer_size: int = 100000, batch_size: int = 64, noise_type: str = 'ou',
                 noise_sigma: float = 0.1, noise_decay: float = 0.9999, device: str = None):
        try:
            from agents.replay_buffer import ReplayBuffer
            from agents.noise import OUNoise, GaussianNoise, DecayingNoise
        except ImportError:
            from .replay_buffer import ReplayBuffer
            from .noise import OUNoise, GaussianNoise, DecayingNoise

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.networks = DDPGNetworks(state_dim, action_dim, hidden_dims, action_low, action_high, device)
        self.actor_optimizer = optim.Adam(self.networks.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.networks.critic.parameters(), lr=critic_lr)
        self.buffer = ReplayBuffer(capacity=buffer_size)

        if noise_type == 'ou':
            base_noise = OUNoise(size=action_dim, sigma=noise_sigma)
        else:
            base_noise = GaussianNoise(size=action_dim, sigma=noise_sigma)
        self.noise = DecayingNoise(base_noise=base_noise, decay_rate=noise_decay, min_scale=0.01)

        self.total_steps = 0
        self.training_stats = {'actor_loss': [], 'critic_loss': [], 'q_values': []}

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        action = self.networks.actor.get_action(state, self.device)
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
        return np.clip(action, self.action_low, self.action_high)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.networks.actor_target(next_states)
            target_q = self.networks.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.networks.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        predicted_actions = self.networks.actor(states)
        actor_loss = -self.networks.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.networks.soft_update(self.tau)

        stats = {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'q_value': current_q.mean().item()}
        self.training_stats['actor_loss'].append(stats['actor_loss'])
        self.training_stats['critic_loss'].append(stats['critic_loss'])
        self.training_stats['q_values'].append(stats['q_value'])
        return stats

    def reset_noise(self):
        self.noise.reset()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'networks': {
                'actor': self.networks.actor.state_dict(),
                'critic': self.networks.critic.state_dict(),
                'actor_target': self.networks.actor_target.state_dict(),
                'critic_target': self.networks.critic_target.state_dict(),
            },
            'optimizers': {'actor': self.actor_optimizer.state_dict(), 'critic': self.critic_optimizer.state_dict()},
            'noise_scale': self.noise.scale,
            'total_steps': self.total_steps,
            'training_stats': self.training_stats,
        }, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.networks.actor.load_state_dict(checkpoint['networks']['actor'])
        self.networks.critic.load_state_dict(checkpoint['networks']['critic'])
        self.networks.actor_target.load_state_dict(checkpoint['networks']['actor_target'])
        self.networks.critic_target.load_state_dict(checkpoint['networks']['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizers']['actor'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizers']['critic'])
        self.noise.scale = checkpoint['noise_scale']
        self.total_steps = checkpoint['total_steps']
        self.training_stats = checkpoint['training_stats']
        print(f"Agent loaded from {filepath}")
