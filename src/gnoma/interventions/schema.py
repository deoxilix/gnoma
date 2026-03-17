"""Intervention ontology schema (T-020).

Defines the canonical data model for molecular interventions
that constitute the RL agent's action space.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class InterventionType(str, Enum):
    """Type of molecular intervention."""

    KNOCKDOWN = "knockdown"  # Gene knockdown (CRISPRi, siRNA)
    OVEREXPRESSION = "overexpression"  # Gene overexpression (CRISPRa, plasmid)
    SMALL_MOLECULE = "small_molecule"  # Pharmacological compound
    EPIGENETIC = "epigenetic"  # Epigenetic modifier (HDAC inhibitor, DNMT inhibitor, etc.)
    REPROGRAMMING_FACTOR = "reprogramming_factor"  # Yamanaka factors, partial reprogramming
    METABOLIC = "metabolic"  # Metabolite supplementation (NAD+, α-KG, etc.)


class OncogenicRisk(str, Enum):
    """Oncogenic risk classification."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    KNOWN_ONCOGENE = "known_oncogene"


class FeasibilityLevel(str, Enum):
    """Real-world therapeutic feasibility."""

    HIGH = "high"  # Approved drug or well-established protocol
    MODERATE = "moderate"  # In clinical trials or advanced preclinical
    LOW = "low"  # Early research stage
    THEORETICAL = "theoretical"  # No existing delivery mechanism


class Intervention(BaseModel):
    """A single molecular intervention in the ontology."""

    id: str = Field(description="Unique intervention identifier, e.g. 'INT-001'")
    name: str = Field(description="Human-readable name")
    type: InterventionType
    target_gene: Optional[str] = Field(
        default=None, description="Primary target gene symbol (HGNC)"
    )
    target_pathway: Optional[str] = Field(
        default=None, description="Primary affected pathway"
    )
    mechanism: str = Field(description="Brief description of molecular mechanism")
    known_aging_effect: Optional[str] = Field(
        default=None, description="Known effect on aging phenotypes from literature"
    )
    oncogenic_risk: OncogenicRisk = Field(default=OncogenicRisk.NONE)
    feasibility: FeasibilityLevel = Field(default=FeasibilityLevel.LOW)
    combinability_restrictions: list[str] = Field(
        default_factory=list,
        description="IDs of interventions that cannot be combined with this one",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Literature references (PMID, DOI, or citation)",
    )
    notes: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


class InterventionOntology(BaseModel):
    """The full intervention ontology — versioned collection of interventions."""

    version: str = Field(description="Ontology version string")
    description: str = Field(default="")
    interventions: list[Intervention] = Field(default_factory=list)

    def get(self, intervention_id: str) -> Intervention:
        """Look up an intervention by ID."""
        for intervention in self.interventions:
            if intervention.id == intervention_id:
                return intervention
        raise KeyError(f"Intervention '{intervention_id}' not found")

    def by_type(self, intervention_type: InterventionType) -> list[Intervention]:
        """Filter interventions by type."""
        return [i for i in self.interventions if i.type == intervention_type]

    def safe_interventions(self) -> list[Intervention]:
        """Return interventions with oncogenic_risk <= LOW."""
        safe_levels = {OncogenicRisk.NONE, OncogenicRisk.LOW}
        return [i for i in self.interventions if i.oncogenic_risk in safe_levels]
