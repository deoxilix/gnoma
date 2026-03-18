"""Evaluation suite for RL agent intervention discovery (T-050).

Measures agent performance on held-out cells, compares against
known rejuvenation benchmarks, and produces candidate reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    n_eval_episodes: int = 100
    top_k_candidates: int = 10
    known_interventions: list[str] = field(
        default_factory=lambda: [
            # Drugs / small molecules
            "rapamycin", "mtor", "metformin", "ampk",
            "nmn", "nad+", "nicotinamide",
            "dasatinib", "quercetin", "senolytic",
            "spermidine", "fisetin", "navitoclax",
            "resveratrol", "sirt",
            # Gene targets
            "foxo3", "klotho", "telomerase", "tert",
            "tfeb", "autophagy",
            "p16", "cdkn2a", "senescence",
            "pgc-1", "ppargc1a", "mitochond",
            "nrf2", "nfe2l2",
            # Reprogramming
            "reprogramming", "oskm", "yamanaka",
            "oct4", "pou5f1", "sox2", "klf4", "myc",
            # Epigenetic
            "dnmt", "ezh2", "hdac", "tet2",
        ]
    )


@dataclass
class EpisodeResult:
    """Result from one evaluation episode."""

    initial_age: float
    final_age: float
    age_delta: float
    total_reward: float
    n_steps: int
    interventions_used: list[str]
    terminated_early: bool
    safety_triggered: bool
    final_hallmarks: dict[str, float]
    reward_components: list[dict[str, float]]


@dataclass
class CandidateIntervention:
    """A ranked intervention candidate from evaluation."""

    rank: int
    intervention_name: str
    intervention_id: str
    frequency: int  # how often the agent selected it
    avg_reward_when_used: float
    avg_age_delta_when_used: float
    confidence: float
    literature_validated: bool


def run_evaluation(
    env,
    agent,
    config: EvalConfig | None = None,
) -> list[EpisodeResult]:
    """Run evaluation episodes and collect results.

    Args:
        env: AgingEnv instance.
        agent: Trained agent with .predict(obs) -> action method.
        config: Eval configuration.

    Returns:
        List of episode results.
    """
    config = config or EvalConfig()
    results = []

    for ep in range(config.n_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        interventions = []
        reward_history = []
        terminated = truncated = False

        while not terminated and not truncated:
            action_masks = env.action_masks()
            action, _ = agent.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            interventions.append(info["intervention"])
            reward_history.append(info["reward_components"])

        result = EpisodeResult(
            initial_age=info.get("initial_age", 0),
            final_age=info.get("age", 0),
            age_delta=info.get("initial_age", 0) - info.get("age", 0),
            total_reward=episode_reward,
            n_steps=info.get("step", 0),
            interventions_used=interventions,
            terminated_early=terminated,
            safety_triggered=any(r.get("safety_triggered", False) for r in reward_history),
            final_hallmarks=info.get("hallmark_scores", {}),
            reward_components=reward_history,
        )
        results.append(result)

    return results


def rank_candidates(
    results: list[EpisodeResult],
    config: EvalConfig | None = None,
) -> list[CandidateIntervention]:
    """Rank interventions by frequency and effectiveness.

    Args:
        results: Evaluation episode results.
        config: Eval configuration.

    Returns:
        Top-K ranked candidate interventions.
    """
    config = config or EvalConfig()
    known = {k.lower() for k in config.known_interventions}

    # Aggregate per-intervention statistics
    intervention_stats: dict[str, dict[str, Any]] = {}

    for result in results:
        for i, name in enumerate(result.interventions_used):
            if name not in intervention_stats:
                intervention_stats[name] = {
                    "count": 0,
                    "rewards": [],
                    "age_deltas": [],
                }
            intervention_stats[name]["count"] += 1
            if i < len(result.reward_components):
                intervention_stats[name]["rewards"].append(result.reward_components[i].get("total", 0))

    # Rank by frequency * mean reward
    ranked = []
    for name, stats in intervention_stats.items():
        avg_reward = float(np.mean(stats["rewards"])) if stats["rewards"] else 0.0
        score = stats["count"] * max(avg_reward, 0)

        ranked.append(
            {
                "name": name,
                "count": stats["count"],
                "avg_reward": avg_reward,
                "score": score,
                "literature_validated": any(k in name.lower() for k in known),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)

    candidates = []
    for i, r in enumerate(ranked[: config.top_k_candidates]):
        candidates.append(
            CandidateIntervention(
                rank=i + 1,
                intervention_name=r["name"],
                intervention_id="",  # filled in by caller if needed
                frequency=r["count"],
                avg_reward_when_used=r["avg_reward"],
                avg_age_delta_when_used=0.0,
                confidence=min(r["count"] / max(len(results), 1), 1.0),
                literature_validated=r["literature_validated"],
            )
        )

    return candidates


def generate_report(
    results: list[EpisodeResult],
    candidates: list[CandidateIntervention],
    output_path: Path | str | None = None,
) -> dict:
    """Generate structured evaluation report.

    Returns:
        Report dict (also saved to JSON if output_path given).
    """
    n_episodes = len(results)
    mean_reward = float(np.mean([r.total_reward for r in results]))
    mean_age_delta = float(np.mean([r.age_delta for r in results]))
    safety_rate = sum(r.safety_triggered for r in results) / max(n_episodes, 1)
    early_term_rate = sum(r.terminated_early for r in results) / max(n_episodes, 1)

    n_validated = sum(1 for c in candidates if c.literature_validated)

    report = {
        "summary": {
            "n_episodes": n_episodes,
            "mean_total_reward": mean_reward,
            "mean_age_delta_years": mean_age_delta,
            "safety_trigger_rate": safety_rate,
            "early_termination_rate": early_term_rate,
        },
        "candidates": [
            {
                "rank": c.rank,
                "name": c.intervention_name,
                "frequency": c.frequency,
                "avg_reward": c.avg_reward_when_used,
                "confidence": c.confidence,
                "literature_validated": c.literature_validated,
            }
            for c in candidates
        ],
        "literature_overlap": {
            "n_validated_in_top_k": n_validated,
            "top_k": len(candidates),
        },
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        logger.info(f"Evaluation report saved to {path}")

    return report
