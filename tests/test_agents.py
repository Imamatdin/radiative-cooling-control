"""
Agent correctness tests (run: python -m pytest tests/test_agents.py -q).

The toy problem is a one-step bandit with a known optimum in environment units: reward = -(alpha - 0.25)^2
- (beta + 0.5)^2, so alpha* = 0.25 and beta* = -0.5. Every agent must find it. An agent that optimises its actor
against critic values at actions in the wrong units (the bug fixed on 2026-10-25) lands far from alpha*.
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from multi_city_training import DDPGAgent, TD3Agent, SACAgent, GaussianActor, to_env_action, set_seed  # noqa: E402

A_STAR = np.array([0.25, -0.5])


def bandit_reward(a):
    return -float(((np.asarray(a) - A_STAR) ** 2).sum())


def test_to_env_action_range_and_gradient():
    raw = torch.tensor([[-1.0, -1.0], [0.0, 0.3], [1.0, 1.0]], requires_grad=True)
    env = to_env_action(raw)
    assert torch.allclose(env, torch.tensor([[0.0, -1.0], [0.5, 0.3], [1.0, 1.0]]))
    env.sum().backward()
    assert torch.allclose(raw.grad, torch.tensor([[0.5, 1.0]] * 3))


def test_sac_log_prob_matches_reference():
    """GaussianActor's tanh-corrected log-prob equals torch.distributions' TransformedDistribution."""
    torch.manual_seed(0)
    actor = GaussianActor(4, 2)
    x = torch.randn(256, 4)
    torch.manual_seed(1)
    a, logp = actor(x)
    h = actor.body(x)
    mu, log_std = actor.mu(h), actor.log_std(h).clamp(actor.LOG_STD_MIN, actor.LOG_STD_MAX)
    dist = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mu, log_std.exp()), [torch.distributions.transforms.TanhTransform(cache_size=1)])
    ref = dist.log_prob(a.clamp(-1 + 1e-6, 1 - 1e-6)).sum(1, keepdim=True)
    ok = (a.abs() < 0.999).all(1)              # atanh is ill-conditioned at the edges; compare the interior
    assert torch.allclose(logp[ok], ref[ok], atol=1e-3)


@pytest.mark.parametrize("cls", [DDPGAgent, TD3Agent, SACAgent])
def test_agent_finds_bandit_optimum(cls):
    set_seed(0)
    agent = cls(state_dim=3, action_dim=2, use_forecast=False)
    state = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    for _ in range(400):                                   # uniform warmup in env units, like train_agent
        a = np.array([np.random.rand(), np.random.uniform(-1, 1)])
        agent.store(state, None, a, bandit_reward(a), state, None, True)
    for _ in range(1500):
        a = agent.select_action(state, None, noise=0.15)
        agent.store(state, None, a, bandit_reward(a), state, None, True)
        agent.train_step()
    got = agent.select_action(state, None, noise=0)
    assert np.abs(got - A_STAR).max() < 0.1, f"{cls.__name__} converged to {got}, optimum {A_STAR}"
