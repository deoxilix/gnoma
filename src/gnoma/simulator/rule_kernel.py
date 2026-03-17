"""Mechanistic rule kernel for the hybrid world model (T-030, T-031).

Encodes curated causal rules mapping interventions to biological
state transitions based on known aging pathway biology.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A single mechanistic rule.

    Maps an intervention (or class of interventions) to an expected
    change in hallmark scores and/or latent state direction.
    """

    id: str
    name: str
    description: str
    intervention_match: dict = field(default_factory=dict)
    """Matching criteria: e.g., {"target_gene": "MTOR", "type": "small_molecule"}
    or {"id": "INT-001"}. Supports partial matching."""
    hallmark_deltas: dict[str, float] = field(default_factory=dict)
    """Expected change in hallmark scores: {hallmark_name: delta}.
    Positive = increase, negative = decrease."""
    confidence: float = 0.8
    """Confidence in this rule (0-1). Used to weight rule vs. neural model."""
    sources: list[str] = field(default_factory=list)


class RuleKernel:
    """Mechanistic rule kernel.

    Applies curated biological rules to predict state transitions.
    Returns hallmark-level deltas that serve as a structured prior
    for the hybrid world model.
    """

    def __init__(self, rules: list[Rule] | None = None):
        self.rules = rules or []
        self._build_index()

    def _build_index(self):
        """Build lookup indices for fast rule matching."""
        self._by_gene: dict[str, list[Rule]] = {}
        self._by_id: dict[str, list[Rule]] = {}
        self._by_type: dict[str, list[Rule]] = {}

        for rule in self.rules:
            match = rule.intervention_match
            if "target_gene" in match:
                self._by_gene.setdefault(match["target_gene"], []).append(rule)
            if "id" in match:
                self._by_id.setdefault(match["id"], []).append(rule)
            if "type" in match:
                self._by_type.setdefault(match["type"], []).append(rule)

    def match_rules(self, intervention: dict) -> list[Rule]:
        """Find all rules matching an intervention.

        Args:
            intervention: Dict with keys like 'id', 'target_gene', 'type'.

        Returns:
            List of matching rules, most specific first.
        """
        matched = []

        # Match by intervention ID (most specific)
        if "id" in intervention and intervention["id"] in self._by_id:
            matched.extend(self._by_id[intervention["id"]])

        # Match by target gene
        if "target_gene" in intervention and intervention["target_gene"] in self._by_gene:
            for rule in self._by_gene[intervention["target_gene"]]:
                if rule not in matched:
                    matched.append(rule)

        # Match by intervention type (least specific)
        if "type" in intervention and intervention["type"] in self._by_type:
            for rule in self._by_type[intervention["type"]]:
                if rule not in matched:
                    matched.append(rule)

        return matched

    def apply(
        self,
        hallmark_scores: dict[str, float],
        intervention: dict,
    ) -> tuple[dict[str, float], float]:
        """Apply matching rules to compute hallmark score deltas.

        Args:
            hallmark_scores: Current hallmark scores.
            intervention: Intervention dict for matching.

        Returns:
            Tuple of (new_hallmark_scores, rule_confidence).
            If no rules match, returns original scores with confidence=0.
        """
        rules = self.match_rules(intervention)

        if not rules:
            return dict(hallmark_scores), 0.0

        # Aggregate deltas from all matching rules (confidence-weighted average)
        aggregated_deltas: dict[str, float] = {}
        total_confidence = 0.0

        for rule in rules:
            for hallmark, delta in rule.hallmark_deltas.items():
                if hallmark in aggregated_deltas:
                    aggregated_deltas[hallmark] += delta * rule.confidence
                else:
                    aggregated_deltas[hallmark] = delta * rule.confidence
            total_confidence += rule.confidence

        if total_confidence > 0:
            for h in aggregated_deltas:
                aggregated_deltas[h] /= total_confidence

        # Apply deltas
        new_scores = dict(hallmark_scores)
        for hallmark, delta in aggregated_deltas.items():
            if hallmark in new_scores:
                new_scores[hallmark] = new_scores[hallmark] + delta

        avg_confidence = total_confidence / len(rules)
        return new_scores, avg_confidence

    @classmethod
    def from_json(cls, path: str | Path) -> RuleKernel:
        """Load rules from JSON file."""
        raw = json.loads(Path(path).read_text())
        rules = [Rule(**r) for r in raw["rules"]]
        return cls(rules)

    def save(self, path: str | Path):
        """Save rules to JSON."""
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"rules": [asdict(r) for r in self.rules]}
        path.write_text(json.dumps(data, indent=2))


def build_default_rules() -> list[Rule]:
    """Build the initial curated rule set for aging pathways."""
    rules = [
        Rule(
            id="RULE-001",
            name="mTOR inhibition reduces senescence",
            description="mTOR inhibition (rapamycin) promotes autophagy and reduces SASP",
            intervention_match={"target_gene": "MTOR"},
            hallmark_deltas={
                "cellular_senescence": -0.3,
                "disabled_macroautophagy": -0.4,
                "deregulated_nutrient_sensing": -0.3,
                "loss_of_proteostasis": -0.2,
            },
            confidence=0.9,
            sources=["PMID:19587680", "PMID:24048020"],
        ),
        Rule(
            id="RULE-002",
            name="AMPK activation mimics caloric restriction",
            description="AMPK activation (metformin) improves nutrient sensing and autophagy",
            intervention_match={"target_gene": "PRKAA1"},
            hallmark_deltas={
                "deregulated_nutrient_sensing": -0.3,
                "disabled_macroautophagy": -0.3,
                "mitochondrial_dysfunction": -0.2,
                "chronic_inflammation": -0.15,
            },
            confidence=0.8,
            sources=["PMID:24931903"],
        ),
        Rule(
            id="RULE-003",
            name="NAD+ restoration improves mitochondrial function",
            description="NAD+ precursors (NMN/NR) restore sirtuin activity and mitochondrial health",
            intervention_match={"type": "metabolic"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.3,
                "deregulated_nutrient_sensing": -0.2,
                "epigenetic_alterations": -0.1,
            },
            confidence=0.7,
            sources=["PMID:27127236"],
        ),
        Rule(
            id="RULE-004",
            name="Senolytics clear senescent cells",
            description="Senolytic compounds (D+Q, Fisetin, Navitoclax) selectively kill senescent cells",
            intervention_match={"target_pathway": "Senescence / SASP"},
            hallmark_deltas={
                "cellular_senescence": -0.5,
                "chronic_inflammation": -0.3,
                "altered_intercellular_communication": -0.2,
                "stem_cell_exhaustion": -0.15,
            },
            confidence=0.85,
            sources=["PMID:25754370", "PMID:30048340"],
        ),
        Rule(
            id="RULE-005",
            name="Partial epigenetic reprogramming reverses aging",
            description="OSKM/OSK resets epigenetic clocks and hallmark scores toward youthful state",
            intervention_match={"type": "reprogramming_factor"},
            hallmark_deltas={
                "epigenetic_alterations": -0.5,
                "cellular_senescence": -0.3,
                "telomere_attrition": -0.2,
                "stem_cell_exhaustion": -0.3,
                "loss_of_proteostasis": -0.1,
            },
            confidence=0.75,
            sources=["PMID:27984723", "PMID:33268865"],
        ),
        Rule(
            id="RULE-006",
            name="SIRT6 overexpression improves genomic stability",
            description="SIRT6 promotes DNA repair, suppresses NF-κB inflammation, maintains telomeres",
            intervention_match={"target_gene": "SIRT6"},
            hallmark_deltas={
                "genomic_instability": -0.3,
                "chronic_inflammation": -0.2,
                "telomere_attrition": -0.2,
                "epigenetic_alterations": -0.15,
            },
            confidence=0.8,
            sources=["PMID:22367546"],
        ),
        Rule(
            id="RULE-007",
            name="Telomerase extends telomeres",
            description="TERT overexpression lengthens telomeres and delays replicative senescence",
            intervention_match={"target_gene": "TERT"},
            hallmark_deltas={
                "telomere_attrition": -0.5,
                "cellular_senescence": -0.2,
                "stem_cell_exhaustion": -0.2,
            },
            confidence=0.8,
            sources=["PMID:22585399"],
        ),
        Rule(
            id="RULE-008",
            name="Spermidine induces autophagy",
            description="Spermidine promotes autophagy via eIF5A hypusination and TFEB activation",
            intervention_match={"id": "INT-010"},
            hallmark_deltas={
                "disabled_macroautophagy": -0.4,
                "loss_of_proteostasis": -0.3,
                "mitochondrial_dysfunction": -0.15,
            },
            confidence=0.75,
            sources=["PMID:19801985"],
        ),
        Rule(
            id="RULE-009",
            name="FOXO3 activation improves stress resistance",
            description="FOXO3 activates autophagy, DNA repair, and antioxidant programs",
            intervention_match={"target_gene": "FOXO3"},
            hallmark_deltas={
                "genomic_instability": -0.2,
                "disabled_macroautophagy": -0.3,
                "loss_of_proteostasis": -0.2,
                "deregulated_nutrient_sensing": -0.15,
            },
            confidence=0.7,
            sources=["PMID:18765803"],
        ),
        Rule(
            id="RULE-010",
            name="Klotho suppresses aging pathways",
            description="Klotho inhibits insulin/IGF-1 signaling, reduces oxidative stress",
            intervention_match={"target_gene": "KL"},
            hallmark_deltas={
                "deregulated_nutrient_sensing": -0.3,
                "chronic_inflammation": -0.2,
                "mitochondrial_dysfunction": -0.15,
                "altered_intercellular_communication": -0.2,
            },
            confidence=0.75,
            sources=["PMID:15731200"],
        ),
        Rule(
            id="RULE-011",
            name="NRF2 activation reduces oxidative damage",
            description="NRF2/KEAP1 pathway activation improves antioxidant defense and proteostasis",
            intervention_match={"target_gene": "NFE2L2"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.2,
                "loss_of_proteostasis": -0.2,
                "genomic_instability": -0.15,
            },
            confidence=0.7,
            sources=["PMID:24299543"],
        ),
        Rule(
            id="RULE-012",
            name="HDAC inhibition opens chromatin",
            description="HDAC inhibitors increase histone acetylation, potentially rejuvenating epigenetic state",
            intervention_match={"type": "epigenetic"},
            hallmark_deltas={
                "epigenetic_alterations": -0.2,
            },
            confidence=0.5,
            sources=["PMID:17182790"],
        ),
        Rule(
            id="RULE-013",
            name="p16 knockdown — senescence bypass with cancer risk",
            description="Loss of p16 reduces senescence but is oncogenic",
            intervention_match={"target_gene": "CDKN2A"},
            hallmark_deltas={
                "cellular_senescence": -0.5,
                "genomic_instability": 0.4,  # increased cancer risk
            },
            confidence=0.9,
            sources=["PMID:22048312"],
        ),
        Rule(
            id="RULE-014",
            name="p53 knockdown — apoptosis bypass with cancer risk",
            description="Loss of p53 prevents apoptosis/senescence but is severely oncogenic",
            intervention_match={"target_gene": "TP53"},
            hallmark_deltas={
                "cellular_senescence": -0.4,
                "genomic_instability": 0.6,  # major cancer risk
            },
            confidence=0.95,
            sources=["PMID:15520276"],
        ),
        Rule(
            id="RULE-015",
            name="TFAM overexpression improves mitochondrial biogenesis",
            description="TFAM drives mtDNA replication and transcription, rescuing age-related decline",
            intervention_match={"target_gene": "TFAM"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.35,
            },
            confidence=0.7,
            sources=["PMID:27065099"],
        ),
        Rule(
            id="RULE-016",
            name="αKG modulates epigenetic landscape via TET enzymes",
            description="Alpha-ketoglutarate is a cofactor for TET demethylases, promoting DNA demethylation",
            intervention_match={"id": "INT-016"},
            hallmark_deltas={
                "epigenetic_alterations": -0.25,
                "deregulated_nutrient_sensing": -0.15,
            },
            confidence=0.65,
            sources=["PMID:24828042", "PMID:32877690"],
        ),
    ]
    return rules
