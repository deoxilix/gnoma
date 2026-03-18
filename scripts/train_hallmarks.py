#!/usr/bin/env python3
"""Train hallmark scoring heads on latent embeddings (T-013, T-014).

Expects:
  - data/processed/{train,val}.h5ad (preprocessed data with expression)
  - data/latent/{latent_train,latent_val}.npy (from train_encoder.py)
  - data/hallmarks/hallmark_gene_sets.json

Outputs: models/hallmarks/ (one .pt per hallmark head)
"""

import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    processed_dir = Path("data/processed")
    latent_dir = Path("data/latent")
    gene_sets_path = Path("data/hallmarks/hallmark_gene_sets.json")
    model_dir = Path("models/hallmarks")

    # Check prerequisites
    for p in [processed_dir / "train.h5ad", latent_dir / "latent_train.npy", gene_sets_path]:
        if not p.exists():
            logger.error(f"Missing: {p}")
            sys.exit(1)

    # Load data
    logger.info("Loading data...")
    adata_train = ad.read_h5ad(processed_dir / "train.h5ad")
    adata_val = ad.read_h5ad(processed_dir / "val.h5ad")
    latent_train = np.load(latent_dir / "latent_train.npy")
    latent_val = np.load(latent_dir / "latent_val.npy")

    # Load hallmark gene sets
    from gnoma.models.hallmark_heads import (
        HallmarkConfig,
        HallmarkScorer,
        compute_gene_set_scores,
        load_hallmark_gene_sets,
    )

    gene_sets = load_hallmark_gene_sets(gene_sets_path)
    logger.info(f"Loaded {len(gene_sets)} hallmark gene sets")

    # Get expression matrices (log-normalized)
    # Census data uses integer var_names; actual gene symbols are in var['feature_name']
    if "feature_name" in adata_train.var.columns:
        gene_names = list(adata_train.var["feature_name"])
    else:
        gene_names = list(adata_train.var_names)
    logger.info(f"Gene name sample: {gene_names[:5]}")
    expr_train = adata_train.X
    expr_val = adata_val.X

    # Convert sparse to dense if needed
    import scipy.sparse

    if scipy.sparse.issparse(expr_train):
        expr_train = expr_train.toarray()
    if scipy.sparse.issparse(expr_val):
        expr_val = expr_val.toarray()

    # Create scorer
    hallmark_names = list(gene_sets.keys())
    config = HallmarkConfig(hidden_dim=64, max_epochs=100, patience=15)
    scorer = HallmarkScorer(hallmark_names, input_dim=latent_train.shape[1], config=config)

    # Train each hallmark head
    all_metrics = {}
    for name in hallmark_names:
        logger.info(f"Training hallmark head: {name}")

        # Compute pseudo-labels from gene set enrichment
        scores_train = compute_gene_set_scores(expr_train, gene_names, gene_sets[name])
        scores_val = compute_gene_set_scores(expr_val, gene_names, gene_sets[name])

        n_matched = sum(1 for g in gene_sets[name] if g in set(gene_names))
        logger.info(f"  Matched {n_matched}/{len(gene_sets[name])} genes")

        if n_matched == 0:
            logger.warning(f"  Skipping {name} — no genes matched")
            continue

        metrics = scorer.train_head(name, latent_train, scores_train, latent_val, scores_val)
        all_metrics[name] = metrics

    # Save
    scorer.save(model_dir)

    # Save metrics
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n=== Hallmark Training Summary ===")
    for name, m in all_metrics.items():
        print(f"  {name}: Spearman ρ={m['spearman_rho']:.3f}")
    print(f"\nSaved to {model_dir}")


if __name__ == "__main__":
    main()
