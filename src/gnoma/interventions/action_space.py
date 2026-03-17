"""RL action space built on the intervention ontology (T-022).

Provides a Gymnasium-compatible discrete action space with
biological safety masking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from gnoma.interventions.schema import (
    InterventionOntology,
    OncogenicRisk,
)

logger = logging.getLogger(__name__)


class InterventionSpace:
    """Discrete action space over curated interventions with safety masking.

    Each action index maps to an intervention in the ontology.
    The mask method returns a boolean array indicating which actions
    are valid given the current state and safety constraints.
    """

    def __init__(
        self,
        ontology: InterventionOntology,
        allow_oncogenic: bool = False,
        embedding_dim: int = 64,
        seed: int = 42,
    ):
        self.ontology = ontology
        self.interventions = ontology.interventions
        self.allow_oncogenic = allow_oncogenic
        self._id_to_idx = {i.id: idx for idx, i in enumerate(self.interventions)}
        self._rng = np.random.RandomState(seed)

        # Pre-compute static safety mask
        self._static_mask = self._compute_static_mask()

        # Initialize embeddings (random for now; will be learned or replaced)
        self._embeddings = self._rng.randn(len(self.interventions), embedding_dim).astype(
            np.float32
        )

    def __len__(self) -> int:
        return len(self.interventions)

    def _compute_static_mask(self) -> np.ndarray:
        """Compute mask based on static properties (oncogenic risk, etc.)."""
        mask = np.ones(len(self.interventions), dtype=bool)
        if not self.allow_oncogenic:
            blocked = {OncogenicRisk.HIGH, OncogenicRisk.KNOWN_ONCOGENE}
            for idx, intervention in enumerate(self.interventions):
                if intervention.oncogenic_risk in blocked:
                    mask[idx] = False
        return mask

    def mask(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Return boolean mask of valid actions.

        Args:
            state: Current biological state vector. Currently unused
                   but will support state-dependent masking (e.g.,
                   blocking interventions incompatible with current
                   cell state).

        Returns:
            Boolean array of shape (n_interventions,).
        """
        # Start with static mask
        m = self._static_mask.copy()

        # TODO: state-dependent masking (Sprint 8)
        # e.g., block interventions on genes already at saturation

        return m

    def sample(self, state: Optional[np.ndarray] = None) -> int:
        """Sample a valid action index uniformly at random."""
        valid = np.where(self.mask(state))[0]
        if len(valid) == 0:
            raise RuntimeError("No valid actions available")
        return int(self._rng.choice(valid))

    def get_intervention(self, action_idx: int):
        """Get the Intervention object for an action index."""
        return self.interventions[action_idx]

    def get_intervention_by_id(self, intervention_id: str):
        """Look up by intervention ID string."""
        return self.ontology.get(intervention_id)

    def to_embedding(self, action_idx: int) -> np.ndarray:
        """Get the embedding vector for an action.

        Returns:
            Float32 array of shape (embedding_dim,).
        """
        return self._embeddings[action_idx]

    def action_summary(self, action_idx: int) -> str:
        """Human-readable summary of an action."""
        i = self.interventions[action_idx]
        return f"[{i.id}] {i.name} ({i.type}) — {i.mechanism[:80]}"

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> InterventionSpace:
        """Load from ontology JSON file."""
        raw = json.loads(Path(path).read_text())
        ontology = InterventionOntology(**raw)
        return cls(ontology, **kwargs)
