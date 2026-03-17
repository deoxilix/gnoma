"""Tests for neural transition model and hybrid world model (T-032, T-033)."""

import numpy as np
import torch

from gnoma.simulator.transition_model import (
    HybridWorldModel,
    TransitionConfig,
    TransitionNetwork,
)


def test_transition_network_forward():
    config = TransitionConfig(latent_dim=32, intervention_embed_dim=16, hidden_dim=64)
    net = TransitionNetwork(config)

    state = torch.randn(4, 32)
    emb = torch.randn(4, 16)
    delta, log_var = net(state, emb)

    assert delta.shape == (4, 32)
    assert log_var.shape == (4, 32)


def test_hybrid_world_model_step():
    config = TransitionConfig(
        latent_dim=32, intervention_embed_dim=16, hidden_dim=64, mc_samples=5
    )
    net = TransitionNetwork(config)
    model = HybridWorldModel(net, config=config)

    state = np.random.randn(32).astype(np.float32)
    emb = np.random.randn(16).astype(np.float32)

    next_state, aleatoric, epistemic = model.step(state, emb)

    assert next_state.shape == (32,)
    assert aleatoric >= 0
    assert epistemic >= 0


def test_step_batch():
    config = TransitionConfig(latent_dim=32, intervention_embed_dim=16, hidden_dim=64)
    net = TransitionNetwork(config)
    model = HybridWorldModel(net, config=config)

    states = np.random.randn(8, 32).astype(np.float32)
    embs = np.random.randn(8, 16).astype(np.float32)

    next_states, aleatoric, epistemic = model.step_batch(states, embs)

    assert next_states.shape == (8, 32)
    assert aleatoric.shape == (8,)
    assert epistemic.shape == (8,)
