#!/usr/bin/env python3
"""Evaluation with diverse intervention discovery.

Runs the PPO agent in stochastic mode (not deterministic) and with
temperature-scaled exploration to discover a broader set of beneficial
interventions. Also includes a greedy baseline for comparison.
"""

import json
import logging
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    # --- Load trained components (same as train_ppo.py) ---
    logger.info("Loading trained components...")

    latent_train = np.load("data/latent/latent_train.npy")
    adata_train = ad.read_h5ad("data/processed/train.h5ad")

    def extract_age(stage):
        match = re.search(r"(\d+)-year", str(stage))
        return int(match.group(1)) if match else None

    if "donor_age" not in adata_train.obs.columns:
        adata_train.obs["donor_age"] = adata_train.obs["development_stage"].apply(extract_age)
    ages = adata_train.obs["donor_age"].values.astype(np.float32)
    valid = ~np.isnan(ages)
    latent_train = latent_train[valid]
    ages = ages[valid]

    from gnoma.models.aging_clock import AgingClockConfig, AgingClockHead

    clock_ckpt = torch.load("models/aging_clock/aging_clock.pt", weights_only=False)
    clock = AgingClockHead(latent_train.shape[1], clock_ckpt["config"])
    clock.load_state_dict(clock_ckpt["state_dict"])
    clock._age_mean = clock_ckpt["age_mean"]
    clock._age_std = clock_ckpt["age_std"]

    gene_sets = json.loads(Path("data/hallmarks/hallmark_gene_sets.json").read_text())
    hallmark_names = list(gene_sets["hallmarks"].keys())
    from gnoma.models.hallmark_heads import HallmarkConfig, HallmarkScorer

    scorer = HallmarkScorer(hallmark_names, input_dim=latent_train.shape[1], config=HallmarkConfig())
    scorer.load("models/hallmarks")

    from gnoma.interventions.action_space import InterventionSpace

    action_space = InterventionSpace.from_json("data/interventions/ontology_v1.json", embedding_dim=64)

    from gnoma.simulator.rule_kernel import build_default_rules, RuleKernel
    from gnoma.simulator.transition_model import TransitionConfig, TransitionNetwork, HybridWorldModel

    rule_kernel = RuleKernel(build_default_rules())

    transition_ckpt_path = Path("models/transition/transition_model.pt")
    if transition_ckpt_path.exists():
        ckpt = torch.load(transition_ckpt_path, weights_only=False)
        trans_config = TransitionConfig(**ckpt["config"])
        trans_config.mc_samples = 5
        trans_net = TransitionNetwork(trans_config)
        trans_net.load_state_dict(ckpt["transition_net_state"])

        import torch.nn as nn
        learned_emb = nn.Embedding(len(ckpt["pert_names"]), trans_config.intervention_embed_dim)
        learned_emb.load_state_dict(ckpt["intervention_embeddings_state"])
        ontology_gene_map = ckpt.get("ontology_gene_map", {})
        for int_id, info in ontology_gene_map.items():
            for idx, intervention in enumerate(action_space.interventions):
                if intervention.id == int_id:
                    action_space._embeddings[idx] = learned_emb.weight[info["pert_idx"]].detach().numpy()
                    break
    else:
        trans_config = TransitionConfig(latent_dim=latent_train.shape[1], intervention_embed_dim=64, hidden_dim=128, mc_samples=5)
        trans_net = TransitionNetwork(trans_config)

    world_model = HybridWorldModel(trans_net, rule_kernel=rule_kernel, config=trans_config)

    from gnoma.reward.reward_fn import RewardConfig, RewardFunction
    from gnoma.env.aging_env import AgingEnv, AgingEnvConfig

    reward_fn = RewardFunction(RewardConfig())
    env_config = AgingEnvConfig(
        latent_dim=latent_train.shape[1],
        n_hallmarks=len(hallmark_names),
        n_interventions=len(action_space),
        max_steps=10,
        initial_age_range=(40.0, 75.0),
    )
    env = AgingEnv(
        world_model=world_model, reward_fn=reward_fn, action_space_obj=action_space,
        hallmark_scorer=scorer, aging_clock=clock, cell_states=latent_train,
        cell_ages=ages, config=env_config,
    )

    # --- Load PPO model ---
    from sb3_contrib import MaskablePPO

    model_path = "models/ppo_agent/ppo_aging_v2"
    if not Path(model_path + ".zip").exists():
        model_path = "models/ppo_agent/ppo_aging_v1"
    model = MaskablePPO.load(model_path)
    logger.info(f"Loaded PPO from {model_path}")

    # --- Evaluation 1: Stochastic mode (diverse discovery) ---
    logger.info("Running stochastic evaluation (200 episodes)...")
    from gnoma.eval.evaluate import EpisodeResult, EvalConfig, rank_candidates, generate_report

    eval_results = []
    for ep in range(200):
        obs, info = env.reset()
        initial_age = info["initial_age"]
        episode_reward = 0.0
        interventions = []
        reward_history = []
        terminated = truncated = False

        while not terminated and not truncated:
            action_mask = env.action_masks()
            # Stochastic: deterministic=False
            action, _ = model.predict(obs, deterministic=False, action_masks=action_mask)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            interventions.append(info["intervention"])
            reward_history.append(info["reward_components"])

        eval_results.append(EpisodeResult(
            initial_age=initial_age,
            final_age=info.get("age", 0),
            age_delta=initial_age - info.get("age", 0),
            total_reward=episode_reward,
            n_steps=info.get("step", 0),
            interventions_used=interventions,
            terminated_early=terminated,
            safety_triggered=any(r.get("safety_triggered", False) for r in reward_history),
            final_hallmarks=info.get("hallmark_scores", {}),
            reward_components=reward_history,
        ))

    config = EvalConfig(n_eval_episodes=200, top_k_candidates=20)
    candidates = rank_candidates(eval_results, config)
    report = generate_report(eval_results, candidates, output_path="reports/eval_ppo_v2_stochastic.json")

    print(f"\n=== Stochastic Evaluation (200 episodes) ===")
    print(f"Mean reward: {report['summary']['mean_total_reward']:.3f}")
    print(f"Mean age delta: {report['summary']['mean_age_delta_years']:.1f} years")
    print(f"Safety trigger rate: {report['summary']['safety_trigger_rate']:.1%}")
    print(f"\nTop interventions:")
    n_unique = len(set(name for r in eval_results for name in r.interventions_used))
    print(f"Unique interventions used: {n_unique}")
    for c in report["candidates"][:20]:
        lit = " [LIT]" if c["literature_validated"] else ""
        print(f"  #{c['rank']} {c['name']}: freq={c['frequency']}, reward={c['avg_reward']:.3f}{lit}")
    print(f"\nLiterature overlap: {report['literature_overlap']['n_validated_in_top_k']}/{report['literature_overlap']['top_k']}")

    # --- Evaluation 2: Random baseline ---
    logger.info("Running random baseline (100 episodes)...")
    random_results = []
    for ep in range(100):
        obs, info = env.reset()
        initial_age = info["initial_age"]
        episode_reward = 0.0
        interventions = []
        reward_history = []
        terminated = truncated = False

        while not terminated and not truncated:
            action = action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            interventions.append(info["intervention"])
            reward_history.append(info["reward_components"])

        random_results.append(EpisodeResult(
            initial_age=initial_age,
            final_age=info.get("age", 0),
            age_delta=initial_age - info.get("age", 0),
            total_reward=episode_reward,
            n_steps=info.get("step", 0),
            interventions_used=interventions,
            terminated_early=terminated,
            safety_triggered=any(r.get("safety_triggered", False) for r in reward_history),
            final_hallmarks=info.get("hallmark_scores", {}),
            reward_components=reward_history,
        ))

    random_reward = np.mean([r.total_reward for r in random_results])
    ppo_reward = report["summary"]["mean_total_reward"]
    improvement = (ppo_reward - random_reward) / abs(random_reward) * 100 if random_reward != 0 else float("inf")

    print(f"\n=== Baseline Comparison ===")
    print(f"Random baseline mean reward: {random_reward:.3f}")
    print(f"PPO stochastic mean reward: {ppo_reward:.3f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Random mean age delta: {np.mean([r.age_delta for r in random_results]):.1f} years")


if __name__ == "__main__":
    main()
