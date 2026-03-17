"""Tests for multi-objective reward function (T-044)."""

import numpy as np

from gnoma.reward.reward_fn import RewardFunction


def test_rejuvenation_positive_for_age_decrease():
    rf = RewardFunction()
    score = rf.rejuvenation_score(age_before=70.0, age_after=65.0)
    assert score > 0


def test_rejuvenation_negative_for_age_increase():
    rf = RewardFunction()
    score = rf.rejuvenation_score(age_before=65.0, age_after=70.0)
    assert score < 0


def test_identity_preservation_no_penalty_when_similar():
    rf = RewardFunction()
    state = np.random.randn(128).astype(np.float32)
    # Small perturbation — high cosine similarity
    perturbed = state + np.random.randn(128).astype(np.float32) * 0.01
    score = rf.identity_preservation(state, perturbed)
    assert score == 0.0


def test_identity_preservation_penalizes_drift():
    rf = RewardFunction()
    state = np.ones(128, dtype=np.float32)
    drifted = -np.ones(128, dtype=np.float32)  # opposite direction
    score = rf.identity_preservation(state, drifted)
    assert score < 0


def test_viability_no_penalty_below_threshold():
    rf = RewardFunction()
    scores = {"genomic_instability": 0.3, "mitochondrial_dysfunction": 0.2}
    assert rf.viability_score(scores) == 0.0


def test_viability_penalizes_high_stress():
    rf = RewardFunction()
    scores = {"genomic_instability": 1.5, "mitochondrial_dysfunction": 0.2}
    assert rf.viability_score(scores) < 0


def test_uncertainty_penalty_zero_below_threshold():
    rf = RewardFunction()
    assert rf.uncertainty_penalty(0.1) == 0.0


def test_uncertainty_penalty_scales_above_threshold():
    rf = RewardFunction()
    p1 = rf.uncertainty_penalty(0.6)
    p2 = rf.uncertainty_penalty(1.0)
    assert p2 > p1 > 0


def test_safety_check_flags_oncogenic():
    rf = RewardFunction()
    safe, _ = rf.safety_check(
        {"genomic_instability": 0.5},
        {"oncogenic_risk": "known_oncogene"},
    )
    assert not safe


def test_safety_check_passes_safe():
    rf = RewardFunction()
    safe, _ = rf.safety_check(
        {"genomic_instability": 0.5},
        {"oncogenic_risk": "none"},
    )
    assert safe


def test_compute_full_reward():
    rf = RewardFunction()
    state = np.random.randn(128).astype(np.float32)
    perturbed = state + np.random.randn(128).astype(np.float32) * 0.05

    reward, components = rf.compute(
        age_before=70.0,
        age_after=68.0,
        initial_state=state,
        current_state=perturbed,
        hallmark_scores={"genomic_instability": 0.3, "mitochondrial_dysfunction": 0.2},
        epistemic_uncertainty=0.1,
    )
    assert isinstance(reward, float)
    assert "rejuvenation" in components
    assert "total" in components
    assert components["total"] == reward
