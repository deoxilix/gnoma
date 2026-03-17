"""Tests for mechanistic rule kernel (T-030, T-031)."""

from gnoma.simulator.rule_kernel import RuleKernel, build_default_rules


def test_build_default_rules():
    rules = build_default_rules()
    assert len(rules) >= 16


def test_rule_matching_by_gene():
    kernel = RuleKernel(build_default_rules())
    matched = kernel.match_rules({"target_gene": "MTOR"})
    assert len(matched) >= 1
    assert any("mTOR" in r.name for r in matched)


def test_rule_matching_by_id():
    kernel = RuleKernel(build_default_rules())
    matched = kernel.match_rules({"id": "INT-010"})
    assert len(matched) >= 1  # spermidine rule


def test_apply_reduces_hallmark_scores():
    kernel = RuleKernel(build_default_rules())
    initial_scores = {
        "cellular_senescence": 0.8,
        "disabled_macroautophagy": 0.7,
        "deregulated_nutrient_sensing": 0.6,
        "loss_of_proteostasis": 0.5,
    }
    new_scores, confidence = kernel.apply(initial_scores, {"target_gene": "MTOR"})
    assert confidence > 0
    assert new_scores["cellular_senescence"] < initial_scores["cellular_senescence"]


def test_apply_no_match_returns_original():
    kernel = RuleKernel(build_default_rules())
    scores = {"cellular_senescence": 0.5}
    new_scores, confidence = kernel.apply(scores, {"target_gene": "NONEXISTENT_GENE"})
    assert confidence == 0.0
    assert new_scores == scores


def test_oncogenic_rules_increase_instability():
    kernel = RuleKernel(build_default_rules())
    scores = {"cellular_senescence": 0.8, "genomic_instability": 0.3}
    new_scores, _ = kernel.apply(scores, {"target_gene": "TP53"})
    assert new_scores["genomic_instability"] > scores["genomic_instability"]
