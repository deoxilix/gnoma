"""Multi-objective reward function with safety constraints (T-044).

Combines rejuvenation progress, identity preservation, viability,
and epistemic uncertainty into a single scalar reward signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Reward function weights and thresholds."""

    # Component weights (must sum to ~1 for interpretability)
    w_rejuvenation: float = 0.5
    w_identity: float = 0.2
    w_viability: float = 0.2
    w_uncertainty: float = 0.1

    # Rejuvenation: aging clock delta improvement
    rejuvenation_scale: float = 1.0

    # Identity preservation: cosine similarity threshold
    identity_min_similarity: float = 0.8

    # Viability: hallmark thresholds for viability proxy
    viability_stress_hallmarks: list[str] = field(
        default_factory=lambda: ["genomic_instability", "mitochondrial_dysfunction"]
    )
    viability_stress_threshold: float = 0.8  # above this = danger

    # Uncertainty penalty
    uncertainty_threshold: float = 0.5  # above this = heavy penalty
    uncertainty_penalty_scale: float = 2.0

    # Safety: hard penalties
    oncogenic_penalty: float = -10.0  # applied if safety filter triggers
    reward_clip: tuple[float, float] = (-5.0, 5.0)


class RewardFunction:
    """Multi-objective reward for the aging RL environment.

    reward = w_rej * rejuvenation_score
           + w_id  * identity_preservation
           + w_via * viability_score
           - w_unc * uncertainty_penalty
           + safety_penalty (if triggered)
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def rejuvenation_score(
        self,
        age_before: float,
        age_after: float,
    ) -> float:
        """Reward for reducing biological age.

        Positive when age decreases. Scaled so 1 year reduction ≈ 0.1 reward.
        """
        delta = age_before - age_after  # positive = rejuvenation
        return delta * self.config.rejuvenation_scale * 0.1

    def identity_preservation(
        self,
        initial_state: np.ndarray,
        current_state: np.ndarray,
    ) -> float:
        """Penalize drift from initial cell identity.

        Uses cosine similarity between initial and current latent states.
        Returns 0 when similarity >= threshold, negative when below.
        """
        norm_init = np.linalg.norm(initial_state)
        norm_curr = np.linalg.norm(current_state)
        if norm_init == 0 or norm_curr == 0:
            return 0.0

        cos_sim = np.dot(initial_state, current_state) / (norm_init * norm_curr)
        threshold = self.config.identity_min_similarity

        if cos_sim >= threshold:
            return 0.0
        else:
            return -(threshold - cos_sim) * 5.0  # scale penalty

    def viability_score(
        self,
        hallmark_scores: dict[str, float],
    ) -> float:
        """Penalize states with dangerously high stress hallmarks.

        Proxy for cell viability — high genomic instability or
        mitochondrial dysfunction suggests non-viable state.
        """
        penalty = 0.0
        threshold = self.config.viability_stress_threshold

        for hallmark in self.config.viability_stress_hallmarks:
            score = hallmark_scores.get(hallmark, 0.0)
            if score > threshold:
                penalty -= (score - threshold) * 3.0

        return penalty

    def uncertainty_penalty(
        self,
        epistemic_uncertainty: float,
    ) -> float:
        """Penalize actions with high epistemic uncertainty.

        Discourages the agent from exploiting poorly-modeled regions
        of the state-action space.
        """
        threshold = self.config.uncertainty_threshold
        if epistemic_uncertainty <= threshold:
            return 0.0

        excess = epistemic_uncertainty - threshold
        return excess * self.config.uncertainty_penalty_scale

    def safety_check(
        self,
        hallmark_scores: dict[str, float],
        intervention_info: dict | None = None,
    ) -> tuple[bool, str]:
        """Check if current state or intervention triggers safety filter.

        Returns:
            (is_safe, reason). If is_safe=False, oncogenic penalty applies.
        """
        # Check oncogenic genomic instability spike
        gi = hallmark_scores.get("genomic_instability", 0.0)
        if gi > 1.5:
            return False, f"genomic_instability={gi:.2f} exceeds safety threshold"

        # Check if intervention itself is flagged
        if intervention_info and intervention_info.get("oncogenic_risk") in (
            "high",
            "known_oncogene",
        ):
            return False, f"intervention has oncogenic_risk={intervention_info['oncogenic_risk']}"

        return True, ""

    def compute(
        self,
        age_before: float,
        age_after: float,
        initial_state: np.ndarray,
        current_state: np.ndarray,
        hallmark_scores: dict[str, float],
        epistemic_uncertainty: float,
        intervention_info: dict | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute full reward.

        Returns:
            (total_reward, component_dict) for logging.
        """
        c = self.config

        r_rej = self.rejuvenation_score(age_before, age_after)
        r_id = self.identity_preservation(initial_state, current_state)
        r_via = self.viability_score(hallmark_scores)
        r_unc = self.uncertainty_penalty(epistemic_uncertainty)

        is_safe, reason = self.safety_check(hallmark_scores, intervention_info)

        total = c.w_rejuvenation * r_rej + c.w_identity * r_id + c.w_viability * r_via - c.w_uncertainty * r_unc

        if not is_safe:
            total += c.oncogenic_penalty
            logger.warning(f"Safety filter triggered: {reason}")

        # Clip
        total = float(np.clip(total, c.reward_clip[0], c.reward_clip[1]))

        components = {
            "rejuvenation": r_rej,
            "identity": r_id,
            "viability": r_via,
            "uncertainty_penalty": r_unc,
            "safety_triggered": not is_safe,
            "total": total,
        }

        return total, components
