"""Tests for intervention action space (T-022)."""

from pathlib import Path

import numpy as np

from gnoma.interventions.action_space import InterventionSpace

ONTOLOGY_PATH = Path(__file__).parent.parent / "data" / "interventions" / "ontology_v1.json"


def test_load_from_json():
    space = InterventionSpace.from_json(ONTOLOGY_PATH)
    assert len(space) == 25


def test_mask_blocks_oncogenic():
    space = InterventionSpace.from_json(ONTOLOGY_PATH, allow_oncogenic=False)
    mask = space.mask()

    # INT-014 (p16 KD), INT-021 (p53 KD), INT-022 (MYC OE) should be blocked
    # They have oncogenic_risk high or known_oncogene
    for idx, intervention in enumerate(space.interventions):
        if intervention.id in ("INT-014", "INT-021", "INT-022"):
            assert not mask[idx], f"{intervention.id} should be masked"


def test_mask_allows_safe():
    space = InterventionSpace.from_json(ONTOLOGY_PATH, allow_oncogenic=False)
    mask = space.mask()

    # INT-001 (Rapamycin) should be allowed
    idx = space._id_to_idx["INT-001"]
    assert mask[idx]


def test_allow_oncogenic_flag():
    space = InterventionSpace.from_json(ONTOLOGY_PATH, allow_oncogenic=True)
    mask = space.mask()
    assert mask.all()


def test_sample_returns_valid_action():
    space = InterventionSpace.from_json(ONTOLOGY_PATH, allow_oncogenic=False)
    for _ in range(50):
        action = space.sample()
        assert space.mask()[action]


def test_embedding_shape():
    space = InterventionSpace.from_json(ONTOLOGY_PATH, embedding_dim=32)
    emb = space.to_embedding(0)
    assert emb.shape == (32,)
    assert emb.dtype == np.float32


def test_action_summary():
    space = InterventionSpace.from_json(ONTOLOGY_PATH)
    summary = space.action_summary(0)
    assert "INT-001" in summary
    assert "Rapamycin" in summary
