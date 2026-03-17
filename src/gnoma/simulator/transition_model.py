"""Neural transition model for the hybrid world model (T-032, T-033, T-034).

Predicts next biological state given current state and intervention.
Integrated with rule kernel to form the hybrid world model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class TransitionConfig:
    """Configuration for neural transition model."""

    latent_dim: int = 128
    intervention_embed_dim: int = 64
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    uncertainty_method: str = "mc_dropout"  # 'mc_dropout' or 'ensemble'
    mc_samples: int = 20
    learning_rate: float = 1e-4
    max_epochs: int = 200
    batch_size: int = 256
    patience: int = 15


class TransitionNetwork(nn.Module):
    """MLP predicting state delta from (state, intervention_embedding).

    Predicts the change in latent state (delta) rather than the
    absolute next state, for easier learning and residual integration.
    """

    def __init__(self, config: TransitionConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim + config.intervention_embed_dim

        layers = []
        prev_dim = input_dim
        for _ in range(config.n_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            prev_dim = config.hidden_dim

        # Predicted state delta
        self.backbone = nn.Sequential(*layers)
        self.delta_head = nn.Linear(config.hidden_dim, config.latent_dim)

        # Aleatoric uncertainty head (log variance)
        self.log_var_head = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, state: torch.Tensor, intervention_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict state delta and aleatoric uncertainty.

        Args:
            state: Current latent state (batch, latent_dim).
            intervention_emb: Intervention embedding (batch, embed_dim).

        Returns:
            (delta, log_var): Predicted delta and log-variance, each (batch, latent_dim).
        """
        x = torch.cat([state, intervention_emb], dim=-1)
        h = self.backbone(x)
        delta = self.delta_head(h)
        log_var = self.log_var_head(h)
        return delta, log_var


class HybridWorldModel:
    """Hybrid world model combining rule kernel and neural transition model.

    The rule kernel provides a structured prior (hallmark-level deltas).
    The neural model predicts residual corrections in the full latent space.
    Uncertainty estimates combine epistemic (MC dropout) and aleatoric components.
    """

    def __init__(
        self,
        transition_net: TransitionNetwork,
        rule_kernel=None,
        config: TransitionConfig | None = None,
    ):
        self.transition_net = transition_net
        self.rule_kernel = rule_kernel
        self.config = config or TransitionConfig()

    def step(
        self,
        state: np.ndarray,
        intervention_emb: np.ndarray,
        intervention_info: dict | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Simulate one transition step.

        Args:
            state: Current latent state (latent_dim,).
            intervention_emb: Intervention embedding (embed_dim,).
            intervention_info: Optional dict for rule kernel matching.

        Returns:
            (next_state, aleatoric_uncertainty, epistemic_uncertainty)
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        emb_t = torch.from_numpy(intervention_emb).float().unsqueeze(0)

        # Neural prediction with MC dropout for epistemic uncertainty
        self.transition_net.train()  # enable dropout
        mc_deltas = []
        mc_log_vars = []

        for _ in range(self.config.mc_samples):
            with torch.no_grad():
                delta, log_var = self.transition_net(state_t, emb_t)
                mc_deltas.append(delta.squeeze(0).numpy())
                mc_log_vars.append(log_var.squeeze(0).numpy())

        self.transition_net.eval()

        mc_deltas = np.stack(mc_deltas)
        mc_log_vars = np.stack(mc_log_vars)

        # Mean prediction and uncertainty
        mean_delta = mc_deltas.mean(axis=0)
        epistemic_unc = float(mc_deltas.var(axis=0).mean())
        aleatoric_unc = float(np.exp(mc_log_vars.mean(axis=0)).mean())

        next_state = state + mean_delta

        return next_state, aleatoric_unc, epistemic_unc

    def step_batch(
        self,
        states: np.ndarray,
        intervention_embs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch transition without MC dropout (for training speed).

        Returns:
            (next_states, aleatoric_unc, epistemic_unc) — epistemic is 0 in non-MC mode.
        """
        self.transition_net.eval()
        states_t = torch.from_numpy(states).float()
        embs_t = torch.from_numpy(intervention_embs).float()

        with torch.no_grad():
            delta, log_var = self.transition_net(states_t, embs_t)

        next_states = states + delta.numpy()
        aleatoric = np.exp(log_var.numpy()).mean(axis=-1)
        epistemic = np.zeros(len(states))

        return next_states, aleatoric, epistemic
