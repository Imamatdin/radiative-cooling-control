"""
Exploration noise for DDPG.
"""

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2, dt: float = 1e-2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu
    
    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()


class GaussianNoise:
    """Simple Gaussian noise for exploration."""
    
    def __init__(self, size: int, sigma: float = 0.1):
        self.size = size
        self.sigma = sigma
    
    def reset(self):
        pass
    
    def sample(self) -> np.ndarray:
        return np.random.randn(self.size) * self.sigma


class DecayingNoise:
    """Wrapper that decays noise over time for annealing exploration."""
    
    def __init__(self, base_noise, decay_rate: float = 0.9999, min_scale: float = 0.01):
        self.base_noise = base_noise
        self.decay_rate = decay_rate
        self.min_scale = min_scale
        self.scale = 1.0
    
    def reset(self):
        self.base_noise.reset()
    
    def sample(self) -> np.ndarray:
        noise = self.base_noise.sample()
        scaled_noise = noise * max(self.scale, self.min_scale)
        self.scale *= self.decay_rate
        return scaled_noise


if __name__ == "__main__":
    print("Testing Exploration Noise...")
    
    ou = OUNoise(size=1, sigma=0.2)
    ou_samples = [ou.sample()[0] for _ in range(100)]
    print(f"OU Noise - Mean: {np.mean(ou_samples):.3f}, Std: {np.std(ou_samples):.3f}")
    
    gauss = GaussianNoise(size=1, sigma=0.2)
    gauss_samples = [gauss.sample()[0] for _ in range(100)]
    print(f"Gaussian Noise - Mean: {np.mean(gauss_samples):.3f}, Std: {np.std(gauss_samples):.3f}")
    
    decaying = DecayingNoise(GaussianNoise(1, 0.2), decay_rate=0.99)
    decay_samples = [decaying.sample()[0] for _ in range(100)]
    print(f"Decaying Noise - Start std: {np.std(decay_samples[:10]):.3f}, End std: {np.std(decay_samples[-10:]):.3f}")
    
    print("\nNoise modules OK!")
