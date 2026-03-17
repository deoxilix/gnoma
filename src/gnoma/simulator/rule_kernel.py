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
        # --- Anti-inflammatory interventions ---
        Rule(
            id="RULE-017",
            name="NF-κB inhibition reduces chronic inflammation",
            description=(
                "Inhibition of NF-κB (e.g., via IKK inhibitors or parthenolide) suppresses "
                "pro-inflammatory cytokine production, reducing inflammaging and SASP-driven "
                "intercellular signaling disruption."
            ),
            intervention_match={"target_gene": "NFKB1"},
            hallmark_deltas={
                "chronic_inflammation": -0.45,
                "altered_intercellular_communication": -0.3,
                "cellular_senescence": -0.1,
            },
            confidence=0.85,
            sources=["PMID:24912023", "PMID:17676033"],
        ),
        Rule(
            id="RULE-018",
            name="JAK/STAT inhibition reduces SASP",
            description=(
                "JAK1/2 inhibitors (e.g., ruxolitinib) block paracrine SASP signaling from "
                "senescent cells, attenuating tissue-wide inflammaging without directly clearing "
                "senescent cells."
            ),
            intervention_match={"target_gene": "JAK1"},
            hallmark_deltas={
                "chronic_inflammation": -0.35,
                "altered_intercellular_communication": -0.3,
                "cellular_senescence": -0.05,
            },
            confidence=0.8,
            sources=["PMID:26829266", "PMID:30082490"],
        ),
        Rule(
            id="RULE-019",
            name="IL-6 neutralization reduces inflammaging",
            description=(
                "Monoclonal antibody neutralization of IL-6 (tocilizumab) or its receptor "
                "suppresses the primary driver of systemic inflammaging, improving tissue "
                "homeostasis and reducing senescence reinforcement via paracrine loops."
            ),
            intervention_match={"target_gene": "IL6"},
            hallmark_deltas={
                "chronic_inflammation": -0.4,
                "altered_intercellular_communication": -0.25,
                "stem_cell_exhaustion": -0.1,
            },
            confidence=0.8,
            sources=["PMID:25828853", "PMID:20562867"],
        ),
        # --- Autophagy modulators ---
        Rule(
            id="RULE-020",
            name="TFEB overexpression improves macroautophagy and proteostasis",
            description=(
                "TFEB (Transcription Factor EB) is the master regulator of lysosomal biogenesis "
                "and autophagy gene expression. Overexpression enhances autophagic flux, clears "
                "aggregated proteins, and improves proteostasis."
            ),
            intervention_match={"target_gene": "TFEB"},
            hallmark_deltas={
                "disabled_macroautophagy": -0.5,
                "loss_of_proteostasis": -0.4,
                "mitochondrial_dysfunction": -0.15,
            },
            confidence=0.82,
            sources=["PMID:23242209", "PMID:21617040"],
        ),
        Rule(
            id="RULE-021",
            name="BECN1 overexpression promotes autophagy",
            description=(
                "Beclin-1 (BECN1) is a core autophagy initiation factor. Overexpression restores "
                "age-related decline in autophagic flux, reduces protein aggregates, and extends "
                "healthspan in model organisms."
            ),
            intervention_match={"target_gene": "BECN1"},
            hallmark_deltas={
                "disabled_macroautophagy": -0.4,
                "loss_of_proteostasis": -0.3,
                "cellular_senescence": -0.1,
            },
            confidence=0.75,
            sources=["PMID:18690243", "PMID:15520276"],
        ),
        Rule(
            id="RULE-022",
            name="Trehalose induces autophagy independently of mTOR",
            description=(
                "Trehalose, a natural disaccharide, induces autophagy through an mTOR-independent "
                "mechanism (TFEB nuclear translocation). It clears misfolded proteins and "
                "aggregate-prone species, improving proteostasis."
            ),
            intervention_match={"id": "INT-022"},
            hallmark_deltas={
                "disabled_macroautophagy": -0.35,
                "loss_of_proteostasis": -0.35,
                "mitochondrial_dysfunction": -0.1,
            },
            confidence=0.7,
            sources=["PMID:23765222", "PMID:28008925"],
        ),
        # --- Mitochondrial interventions ---
        Rule(
            id="RULE-023",
            name="PGC1α/PPARGC1A overexpression drives mitochondrial biogenesis",
            description=(
                "PGC-1α is the master co-activator of mitochondrial biogenesis. Overexpression "
                "increases mitochondrial mass, improves oxidative phosphorylation efficiency, "
                "and reduces ROS production in aged tissues."
            ),
            intervention_match={"target_gene": "PPARGC1A"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.45,
                "deregulated_nutrient_sensing": -0.2,
                "cellular_senescence": -0.1,
            },
            confidence=0.82,
            sources=["PMID:12610026", "PMID:25174007"],
        ),
        Rule(
            id="RULE-024",
            name="PINK1 overexpression enhances mitophagy and mitochondrial quality control",
            description=(
                "PINK1 kinase recruits Parkin to damaged mitochondria, triggering mitophagy. "
                "Overexpression improves selective clearance of dysfunctional mitochondria, "
                "reducing ROS and improving overall mitochondrial network quality."
            ),
            intervention_match={"target_gene": "PINK1"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.4,
                "disabled_macroautophagy": -0.2,
                "genomic_instability": -0.1,
            },
            confidence=0.78,
            sources=["PMID:18443288", "PMID:28178239"],
        ),
        Rule(
            id="RULE-025",
            name="MitoQ/SS-31 reduces mitochondrial ROS",
            description=(
                "Mitochondria-targeted antioxidants (MitoQ, SS-31/elamipretide) accumulate in "
                "the mitochondrial matrix and inner membrane, quenching ROS at the source, "
                "preserving mtDNA integrity and membrane potential."
            ),
            intervention_match={"type": "mito_antioxidant"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.35,
                "genomic_instability": -0.15,
                "cellular_senescence": -0.1,
                "loss_of_proteostasis": -0.1,
            },
            confidence=0.72,
            sources=["PMID:16271456", "PMID:21276366"],
        ),
        # --- Stem cell rejuvenation ---
        Rule(
            id="RULE-026",
            name="Wnt pathway modulation supports stem cell maintenance",
            description=(
                "Balanced Wnt signaling (neither constitutively active nor inactive) is required "
                "for stem cell self-renewal. Age-appropriate Wnt restoration reduces stem cell "
                "exhaustion and improves tissue regenerative capacity."
            ),
            intervention_match={"target_pathway": "Wnt / stem cell"},
            hallmark_deltas={
                "stem_cell_exhaustion": -0.35,
                "altered_intercellular_communication": -0.2,
                "epigenetic_alterations": -0.1,
            },
            confidence=0.7,
            sources=["PMID:18485877", "PMID:23426260"],
        ),
        Rule(
            id="RULE-027",
            name="GDF11 promotes stem cell and tissue rejuvenation",
            description=(
                "GDF11 (Growth Differentiation Factor 11), a circulating TGF-β family member, "
                "declines with age. Restoration improves muscle stem cell function, cardiac "
                "hypertrophy reversal, and neurogenesis in aged animals."
            ),
            intervention_match={"target_gene": "GDF11"},
            hallmark_deltas={
                "stem_cell_exhaustion": -0.4,
                "mitochondrial_dysfunction": -0.15,
                "altered_intercellular_communication": -0.2,
                "epigenetic_alterations": -0.1,
            },
            confidence=0.65,
            sources=["PMID:24797481", "PMID:25236977"],
        ),
        # --- DNA repair ---
        Rule(
            id="RULE-028",
            name="ATM activation improves genomic stability",
            description=(
                "ATM kinase is the master sensor of DNA double-strand breaks. Pharmacological "
                "activation or expression restoration enhances DNA damage response, reduces "
                "mutation accumulation, and suppresses genomic instability in aged cells."
            ),
            intervention_match={"target_gene": "ATM"},
            hallmark_deltas={
                "genomic_instability": -0.35,
                "cellular_senescence": -0.1,
                "epigenetic_alterations": -0.1,
            },
            confidence=0.75,
            sources=["PMID:16041387", "PMID:25774008"],
        ),
        Rule(
            id="RULE-029",
            name="PARP1 modulation enhances DNA repair",
            description=(
                "PARP1 is a key enzyme in single-strand break repair. Careful modulation "
                "(activation at physiological levels) restores efficient DNA repair, while "
                "over-activation depletes NAD+. Net benefit at restoration doses is reduced "
                "genomic instability."
            ),
            intervention_match={"target_gene": "PARP1"},
            hallmark_deltas={
                "genomic_instability": -0.25,
                "epigenetic_alterations": -0.1,
                "mitochondrial_dysfunction": -0.05,
            },
            confidence=0.68,
            sources=["PMID:22014574", "PMID:25574716"],
        ),
        # --- Sirtuin-specific ---
        Rule(
            id="RULE-030",
            name="SIRT1 activation improves nutrient sensing and epigenetic maintenance",
            description=(
                "SIRT1 deacetylates histones and transcription factors (p53, FOXO, NF-κB), "
                "improving epigenetic fidelity and metabolic sensing. Activation by resveratrol "
                "or SRT2104 mimics caloric restriction benefits."
            ),
            intervention_match={"target_gene": "SIRT1"},
            hallmark_deltas={
                "epigenetic_alterations": -0.3,
                "deregulated_nutrient_sensing": -0.25,
                "chronic_inflammation": -0.2,
                "mitochondrial_dysfunction": -0.1,
            },
            confidence=0.78,
            sources=["PMID:15604409", "PMID:23595621"],
        ),
        Rule(
            id="RULE-031",
            name="SIRT3 overexpression improves mitochondrial deacetylation",
            description=(
                "SIRT3 is the primary mitochondrial deacetylase. Overexpression reduces "
                "hyperacetylation of OXPHOS complexes and metabolic enzymes in aged mitochondria, "
                "improving ATP production and reducing ROS."
            ),
            intervention_match={"target_gene": "SIRT3"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.4,
                "deregulated_nutrient_sensing": -0.2,
                "genomic_instability": -0.1,
            },
            confidence=0.78,
            sources=["PMID:21076238", "PMID:21907141"],
        ),
        Rule(
            id="RULE-032",
            name="SIRT7 maintains rDNA stability and nucleolar integrity",
            description=(
                "SIRT7 deacetylates H3K18 at rDNA repeats and interacts with RNA Pol I, "
                "maintaining nucleolar integrity. Age-related SIRT7 decline leads to rDNA "
                "instability and increased epigenetic noise."
            ),
            intervention_match={"target_gene": "SIRT7"},
            hallmark_deltas={
                "genomic_instability": -0.25,
                "epigenetic_alterations": -0.3,
                "cellular_senescence": -0.1,
            },
            confidence=0.7,
            sources=["PMID:26158249", "PMID:23273981"],
        ),
        # --- Dangerous interventions (safety filter testing) ---
        Rule(
            id="RULE-033",
            name="RAS overexpression increases genomic instability (oncogenic)",
            description=(
                "Constitutively active RAS (KRAS G12D, HRAS G12V) drives oncogene-induced "
                "senescence acutely but promotes genomic instability, aneuploidy, and malignant "
                "transformation in permissive contexts. Strongly oncogenic."
            ),
            intervention_match={"target_gene": "KRAS"},
            hallmark_deltas={
                "genomic_instability": 0.65,
                "cellular_senescence": 0.2,  # initial OIS
                "deregulated_nutrient_sensing": 0.3,
            },
            confidence=0.92,
            sources=["PMID:9126741", "PMID:17898714"],
        ),
        Rule(
            id="RULE-034",
            name="AKT overexpression drives proliferation with cancer risk (oncogenic)",
            description=(
                "Constitutively active AKT1/AKT2 overrides apoptotic checkpoints, promotes "
                "hyperproliferative signaling, suppresses FOXO activity, and elevates cancer "
                "risk across tissue types."
            ),
            intervention_match={"target_gene": "AKT1"},
            hallmark_deltas={
                "genomic_instability": 0.4,
                "deregulated_nutrient_sensing": 0.35,
                "cellular_senescence": -0.25,  # bypasses senescence
                "altered_intercellular_communication": 0.2,
            },
            confidence=0.9,
            sources=["PMID:11099048", "PMID:21340308"],
        ),
        Rule(
            id="RULE-035",
            name="BCL2 overexpression blocks apoptosis (oncogenic)",
            description=(
                "BCL2 overexpression inhibits mitochondrial apoptosis, allowing survival of "
                "damaged and pre-malignant cells. While used therapeutically in B-cell lymphoma "
                "contexts, constitutive overexpression increases cancer risk and senescent cell "
                "burden by blocking clearance."
            ),
            intervention_match={"target_gene": "BCL2"},
            hallmark_deltas={
                "genomic_instability": 0.35,
                "cellular_senescence": 0.3,  # accumulates SASP-secreting cells
                "chronic_inflammation": 0.2,
                "altered_intercellular_communication": 0.15,
            },
            confidence=0.88,
            sources=["PMID:2874511", "PMID:22901802"],
        ),
        # --- Metabolic interventions ---
        Rule(
            id="RULE-036",
            name="Ketone body supplementation improves mitochondrial function and nutrient sensing",
            description=(
                "Exogenous ketones (beta-hydroxybutyrate, BHB) or ketogenic diet provide an "
                "alternative fuel substrate, inhibit NLRP3 inflammasome, act as HDAC inhibitors "
                "via BHB-mediated histone modification, and improve mitochondrial efficiency."
            ),
            intervention_match={"type": "ketone_supplement"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.25,
                "deregulated_nutrient_sensing": -0.2,
                "chronic_inflammation": -0.2,
                "epigenetic_alterations": -0.1,
            },
            confidence=0.7,
            sources=["PMID:26312686", "PMID:25686106"],
        ),
        Rule(
            id="RULE-037",
            name="Taurine supplementation improves multiple hallmarks",
            description=(
                "Taurine, a semi-essential sulfur amino acid, declines with age. Restoration "
                "improves mitochondrial function, reduces DNA damage, suppresses cellular "
                "senescence, and extends healthspan in multiple model organisms. Acts broadly "
                "across hallmarks."
            ),
            intervention_match={"id": "INT-037"},
            hallmark_deltas={
                "mitochondrial_dysfunction": -0.25,
                "cellular_senescence": -0.2,
                "genomic_instability": -0.15,
                "stem_cell_exhaustion": -0.15,
                "epigenetic_alterations": -0.1,
                "disabled_macroautophagy": -0.1,
            },
            confidence=0.72,
            sources=["PMID:37272891", "PMID:35595856"],
        ),
    ]
    return rules
