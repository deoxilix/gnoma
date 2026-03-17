"""Experiment tracking utilities (T-061).

Thin wrapper around W&B for standardized experiment logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Standard experiment tracking configuration."""

    project: str = "gnoma"
    entity: Optional[str] = None
    experiment_name: str = "unnamed"
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    config: dict[str, Any] = field(default_factory=dict)


def init_tracking(exp_config: ExperimentConfig, mode: str = "online"):
    """Initialize W&B run with standard gnoma configuration.

    Args:
        exp_config: Experiment configuration.
        mode: W&B mode — 'online', 'offline', or 'disabled'.

    Returns:
        wandb.Run instance (or None if disabled).
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; tracking disabled")
        return None

    run = wandb.init(
        project=exp_config.project,
        entity=exp_config.entity,
        name=exp_config.experiment_name,
        tags=exp_config.tags,
        notes=exp_config.notes,
        config=exp_config.config,
        mode=mode,
    )

    logger.info(f"W&B run initialized: {run.name} ({run.id})")
    return run


def log_metrics(metrics: dict[str, Any], step: Optional[int] = None):
    """Log metrics to active W&B run."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def log_artifact(name: str, artifact_type: str, path: str, metadata: Optional[dict] = None):
    """Log a versioned artifact (dataset, model checkpoint, etc.)."""
    try:
        import wandb

        if wandb.run is None:
            logger.warning("No active W&B run; skipping artifact logging")
            return

        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_reference(f"file://{path}")
        wandb.run.log_artifact(artifact)
        logger.info(f"Logged artifact: {name} ({artifact_type})")
    except ImportError:
        pass
