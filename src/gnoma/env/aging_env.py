"""Gymnasium environment wrapping the aging world model (T-040).

The agent observes a compressed biological state and selects
molecular interventions to rejuvenate aged cells.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass
class AgingEnvConfig:
    """Environment configuration."""

    latent_dim: int = 128
    n_hallmarks: int = 11
    n_interventions: int = 25
    max_steps: int = 10
    viability_threshold: float = -2.0  # terminate if viability reward drops below
    initial_age_range: tuple[float, float] = (50.0, 80.0)
    seed: int = 42


class AgingEnv(gym.Env):
    """Gymnasium environment for cellular rejuvenation via RL.

    Observation: [latent_state | hallmark_scores | predicted_age | step]
    Action: Discrete index into intervention ontology.
    Reward: Multi-objective (rejuvenation + identity + viability - uncertainty).
    Termination: max_steps reached OR viability drops too low.

    Requires external components (world model, reward function, action space,
    hallmark scorer, aging clock) to be injected at construction time.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        world_model,
        reward_fn,
        action_space_obj,
        hallmark_scorer,
        aging_clock,
        cell_states: np.ndarray,
        cell_ages: np.ndarray,
        config: AgingEnvConfig | None = None,
        render_mode: str | None = None,
    ):
        """
        Args:
            world_model: HybridWorldModel with .step() API.
            reward_fn: RewardFunction with .compute() API.
            action_space_obj: InterventionSpace with .mask(), .to_embedding(), etc.
            hallmark_scorer: HallmarkScorer with .score_hallmarks() API.
            aging_clock: AgingClockHead with .predict_age() API.
            cell_states: Pool of initial latent states to sample from (n_cells, latent_dim).
            cell_ages: Corresponding ages (n_cells,).
            config: Environment config.
        """
        super().__init__()

        self.world_model = world_model
        self.reward_fn = reward_fn
        self.intervention_space = action_space_obj
        self.hallmark_scorer = hallmark_scorer
        self.aging_clock = aging_clock
        self.cell_states = cell_states
        self.cell_ages = cell_ages
        self.config = config or AgingEnvConfig()
        self.render_mode = render_mode

        n_int = len(action_space_obj)
        obs_dim = self.config.latent_dim + self.config.n_hallmarks + 2  # +age +step

        # Gymnasium spaces
        self.action_space = spaces.Discrete(n_int)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Episode state
        self._state: Optional[np.ndarray] = None
        self._initial_state: Optional[np.ndarray] = None
        self._current_age: float = 0.0
        self._initial_age: float = 0.0
        self._step_count: int = 0
        self._rng = np.random.RandomState(self.config.seed)

    def _build_obs(self) -> np.ndarray:
        """Build observation vector from current state."""
        hallmarks = self.hallmark_scorer.score_hallmarks(self._state)
        hallmark_vec = np.array([hallmarks[k] for k in sorted(hallmarks.keys())], dtype=np.float32)
        obs = np.concatenate(
            [
                self._state.astype(np.float32),
                hallmark_vec,
                np.array([self._current_age], dtype=np.float32),
                np.array([self._step_count / self.config.max_steps], dtype=np.float32),
            ]
        )
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset environment by sampling a new aged cell."""
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.RandomState(seed)

        # Sample a cell within the target age range
        age_min, age_max = self.config.initial_age_range
        eligible = np.where((self.cell_ages >= age_min) & (self.cell_ages <= age_max))[0]

        if len(eligible) == 0:
            eligible = np.arange(len(self.cell_states))

        idx = self._rng.choice(eligible)
        self._state = self.cell_states[idx].copy()
        self._initial_state = self._state.copy()
        self._initial_age = float(self.cell_ages[idx])
        self._current_age = self._initial_age
        self._step_count = 0

        obs = self._build_obs()
        info = {
            "initial_age": self._initial_age,
            "cell_index": int(idx),
        }

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply an intervention and advance the simulation.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        intervention = self.intervention_space.get_intervention(action)
        intervention_emb = self.intervention_space.to_embedding(action)
        intervention_info = {
            "id": intervention.id,
            "name": intervention.name,
            "type": intervention.type,
            "oncogenic_risk": intervention.oncogenic_risk,
            "target_gene": intervention.target_gene,
        }

        # World model step
        age_before = self._current_age
        next_state, aleatoric_unc, epistemic_unc = self.world_model.step(
            self._state,
            intervention_emb,
            intervention_info=intervention_info,
        )

        self._state = next_state
        self._current_age = float(self.aging_clock.predict_age(next_state[np.newaxis, :])[0])
        self._step_count += 1

        # Compute hallmark scores for reward
        hallmark_scores = self.hallmark_scorer.score_hallmarks(self._state)

        # Compute reward
        reward, reward_components = self.reward_fn.compute(
            age_before=age_before,
            age_after=self._current_age,
            initial_state=self._initial_state,
            current_state=self._state,
            hallmark_scores=hallmark_scores,
            epistemic_uncertainty=epistemic_unc,
            intervention_info=intervention_info,
        )

        # Check termination
        terminated = False
        truncated = self._step_count >= self.config.max_steps

        # Viability termination
        if reward_components["viability"] < self.config.viability_threshold:
            terminated = True

        # Safety termination
        if reward_components["safety_triggered"]:
            terminated = True

        obs = self._build_obs()
        info = {
            "age": self._current_age,
            "age_delta": age_before - self._current_age,
            "step": self._step_count,
            "intervention": intervention.name,
            "reward_components": reward_components,
            "epistemic_uncertainty": epistemic_unc,
            "aleatoric_uncertainty": aleatoric_unc,
            "hallmark_scores": hallmark_scores,
        }

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions (for masked PPO)."""
        return self.intervention_space.mask(self._state)
