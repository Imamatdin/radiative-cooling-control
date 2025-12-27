"""
Experience Replay Buffer for DDPG.
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, priority=None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[i] for i in indices]
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    print("Testing Replay Buffer...")
    
    buffer = ReplayBuffer(capacity=1000)
    for i in range(100):
        state = np.random.randn(5)
        action = np.random.rand(1)
        reward = np.random.randn()
        next_state = np.random.randn(5)
        done = i % 20 == 0
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"Batch shapes: States: {states.shape}, Actions: {actions.shape}")
    print("\nReplay Buffer OK!")
