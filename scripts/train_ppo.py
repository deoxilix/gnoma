#!/usr/bin/env python3
"""Train PPO agent on AgingEnv with trained components (Phase 3).

Wires up: encoder latents + aging clock + hallmark scorer + world model + reward
into the Gymnasium environment and trains a masked PPO agent.

Expects:
  - data/latent/{latent_train,latent_val}.npy
  - data/processed/train.h5ad (for ages)
  - models/aging_clock/aging_clock.pt
  - models/hallmarks/*.pt
  - data/interventions/ontology_v1.json
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    # --- Load trained components ---
    logger.info("Loading trained components...")

    # Latent embeddings and ages
    latent_train = np.load("data/latent/latent_train.npy")
    import anndata as ad

    adata_train = ad.read_h5ad("data/processed/train.h5ad")

    def extract_age(stage):
        match = re.search(r"(\d+)-year", str(stage))
        return int(match.group(1)) if match else None

    if "donor_age" not in adata_train.obs.columns:
        adata_train.obs["donor_age"] = adata_train.obs["development_stage"].apply(extract_age)
    ages = adata_train.obs["donor_age"].values.astype(np.float32)

    # Filter out NaN ages
    valid = ~np.isnan(ages)
    latent_train = latent_train[valid]
    ages = ages[valid]
    logger.info(f"Loaded {len(latent_train)} cells with valid ages")

    # Aging clock
    from gnoma.models.aging_clock import AgingClockConfig, AgingClockHead

    clock_ckpt = torch.load("models/aging_clock/aging_clock.pt", weights_only=False)
    clock_config = clock_ckpt["config"]
    clock = AgingClockHead(latent_train.shape[1], clock_config)
    clock.load_state_dict(clock_ckpt["state_dict"])
    clock._age_mean = clock_ckpt["age_mean"]
    clock._age_std = clock_ckpt["age_std"]
    logger.info(f"Aging clock loaded (R²={clock_ckpt['metrics']['r2']:.3f})")

    # Hallmark scorer
    gene_sets = json.loads(Path("data/hallmarks/hallmark_gene_sets.json").read_text())
    hallmark_names = list(gene_sets["hallmarks"].keys())
    from gnoma.models.hallmark_heads import HallmarkConfig, HallmarkScorer

    scorer = HallmarkScorer(hallmark_names, input_dim=latent_train.shape[1], config=HallmarkConfig())
    scorer.load("models/hallmarks")
    logger.info(f"Hallmark scorer loaded ({len(hallmark_names)} hallmarks)")

    # Intervention space
    from gnoma.interventions.action_space import InterventionSpace

    action_space = InterventionSpace.from_json("data/interventions/ontology_v1.json", embedding_dim=64)
    logger.info(f"Intervention space: {len(action_space)} interventions")

    # World model (rule kernel + trained neural transition model)
    from gnoma.simulator.rule_kernel import build_default_rules, RuleKernel
    from gnoma.simulator.transition_model import TransitionConfig, TransitionNetwork, HybridWorldModel

    rule_kernel = RuleKernel(build_default_rules())

    transition_ckpt_path = Path("models/transition/transition_model.pt")
    if transition_ckpt_path.exists():
        logger.info("Loading trained transition model...")
        ckpt = torch.load(transition_ckpt_path, weights_only=False)
        trans_config = TransitionConfig(**ckpt["config"])
        trans_config.mc_samples = 5  # reduce for env speed
        trans_net = TransitionNetwork(trans_config)
        trans_net.load_state_dict(ckpt["transition_net_state"])
        logger.info(f"Transition model loaded (metrics: centroid_corr={ckpt['metrics']['mean_perturbation_centroid_correlation']:.3f})")

        # Update action space embeddings with learned perturbation embeddings
        import torch.nn as nn
        learned_emb = nn.Embedding(len(ckpt["pert_names"]), trans_config.intervention_embed_dim)
        learned_emb.load_state_dict(ckpt["intervention_embeddings_state"])
        ontology_gene_map = ckpt.get("ontology_gene_map", {})

        # Map learned embeddings to our action space
        for int_id, info in ontology_gene_map.items():
            # Find the action index for this intervention
            for idx, intervention in enumerate(action_space.interventions):
                if intervention.id == int_id:
                    emb_vec = learned_emb.weight[info["pert_idx"]].detach().numpy()
                    action_space._embeddings[idx] = emb_vec
                    logger.info(f"  Loaded learned embedding for {int_id} ({info['gene']})")
                    break
    else:
        logger.info("No trained transition model found — using untrained neural model")
        trans_config = TransitionConfig(
            latent_dim=latent_train.shape[1],
            intervention_embed_dim=64,
            hidden_dim=128,
            mc_samples=5,
        )
        trans_net = TransitionNetwork(trans_config)

    world_model = HybridWorldModel(trans_net, rule_kernel=rule_kernel, config=trans_config)
    logger.info("World model initialized (rule kernel + neural transition)")

    # Reward function
    from gnoma.reward.reward_fn import RewardConfig, RewardFunction

    reward_fn = RewardFunction(RewardConfig())

    # --- Build environment ---
    from gnoma.env.aging_env import AgingEnv, AgingEnvConfig

    env_config = AgingEnvConfig(
        latent_dim=latent_train.shape[1],
        n_hallmarks=len(hallmark_names),
        n_interventions=len(action_space),
        max_steps=10,
        initial_age_range=(40.0, 75.0),
    )

    env = AgingEnv(
        world_model=world_model,
        reward_fn=reward_fn,
        action_space_obj=action_space,
        hallmark_scorer=scorer,
        aging_clock=clock,
        cell_states=latent_train,
        cell_ages=ages,
        config=env_config,
    )

    # Quick sanity check
    obs, info = env.reset()
    logger.info(f"Env observation shape: {obs.shape}, initial age: {info['initial_age']:.1f}")

    # --- Train PPO ---
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    def mask_fn(env):
        return env.action_masks()

    masked_env = ActionMasker(env, mask_fn)

    logger.info("Initializing MaskablePPO (v2 — trained transition model)...")
    model = MaskablePPO(
        "MlpPolicy",
        masked_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,  # higher entropy for more exploration
        seed=42,
    )

    total_timesteps = 100_000
    logger.info(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model_path = Path("models/ppo_agent")
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path / "ppo_aging_v2"))
    logger.info(f"PPO v2 agent saved to {model_path}")

    # --- Quick evaluation ---
    logger.info("Running evaluation (50 episodes)...")
    from gnoma.eval.evaluate import EvalConfig, run_evaluation, rank_candidates, generate_report

    eval_config = EvalConfig(n_eval_episodes=50)

    # Reset env for eval
    eval_results = []
    for ep in range(eval_config.n_eval_episodes):
        obs, info = env.reset()
        initial_age = info["initial_age"]
        episode_reward = 0.0
        interventions = []
        reward_history = []
        terminated = truncated = False

        while not terminated and not truncated:
            action_mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            interventions.append(info["intervention"])
            reward_history.append(info["reward_components"])

        from gnoma.eval.evaluate import EpisodeResult

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

    candidates = rank_candidates(eval_results, eval_config)
    report = generate_report(eval_results, candidates, output_path="reports/eval_ppo_v2.json")

    print(f"\n=== PPO Evaluation Summary ===")
    print(f"Episodes: {report['summary']['n_episodes']}")
    print(f"Mean reward: {report['summary']['mean_total_reward']:.3f}")
    print(f"Mean age delta: {report['summary']['mean_age_delta_years']:.1f} years")
    print(f"Safety trigger rate: {report['summary']['safety_trigger_rate']:.1%}")
    print(f"\nTop interventions:")
    for c in report["candidates"][:10]:
        lit = " [LIT]" if c["literature_validated"] else ""
        print(f"  #{c['rank']} {c['name']}: freq={c['frequency']}, reward={c['avg_reward']:.3f}{lit}")
    print(f"\nLiterature overlap: {report['literature_overlap']['n_validated_in_top_k']}/{report['literature_overlap']['top_k']}")


if __name__ == "__main__":
    main()
