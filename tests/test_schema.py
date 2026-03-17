"""Tests for intervention ontology schema (T-020)."""

import json
from pathlib import Path

from gnoma.interventions.schema import (
    Intervention,
    InterventionOntology,
    InterventionType,
    OncogenicRisk,
)

ONTOLOGY_PATH = Path(__file__).parent.parent / "data" / "interventions" / "ontology_v1.json"


def test_ontology_loads():
    """Ontology JSON loads and validates against schema."""
    raw = json.loads(ONTOLOGY_PATH.read_text())
    ontology = InterventionOntology(**raw)
    assert len(ontology.interventions) >= 25


def test_intervention_lookup():
    raw = json.loads(ONTOLOGY_PATH.read_text())
    ontology = InterventionOntology(**raw)
    rapamycin = ontology.get("INT-001")
    assert rapamycin.name == "Rapamycin (mTOR inhibition)"
    assert rapamycin.type == InterventionType.SMALL_MOLECULE


def test_safe_interventions_excludes_oncogenes():
    raw = json.loads(ONTOLOGY_PATH.read_text())
    ontology = InterventionOntology(**raw)
    safe = ontology.safe_interventions()
    for i in safe:
        assert i.oncogenic_risk in (OncogenicRisk.NONE, OncogenicRisk.LOW), (
            f"{i.id} ({i.name}) has oncogenic_risk={i.oncogenic_risk}"
        )
    # p16 KD and p53 KD and MYC OE should be excluded
    safe_ids = {i.id for i in safe}
    assert "INT-014" not in safe_ids  # p16 KD
    assert "INT-021" not in safe_ids  # p53 KD
    assert "INT-022" not in safe_ids  # MYC OE


def test_by_type():
    raw = json.loads(ONTOLOGY_PATH.read_text())
    ontology = InterventionOntology(**raw)
    senolytics = ontology.by_type(InterventionType.SMALL_MOLECULE)
    assert len(senolytics) >= 5


def test_intervention_schema_validation():
    """Test that a minimal intervention passes validation."""
    i = Intervention(
        id="TEST-001",
        name="Test intervention",
        type=InterventionType.KNOCKDOWN,
        mechanism="Test mechanism",
    )
    assert i.id == "TEST-001"
    assert i.oncogenic_risk == OncogenicRisk.NONE
