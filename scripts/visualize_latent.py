#!/usr/bin/env python3
"""Visualize latent space with UMAP colored by age, cell type, donor (Phase 1 validation).

Expects data/latent/latent_train.npy and data/processed/train.h5ad.
Outputs figures to figures/.
"""

import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import umap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    latent = np.load("data/latent/latent_train.npy")
    adata = ad.read_h5ad("data/processed/train.h5ad")
    logger.info(f"Latent: {latent.shape}, Cells: {adata.n_obs}")

    # Subsample for UMAP speed
    n = min(10_000, len(latent))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(latent), n, replace=False)
    latent_sub = latent[idx]
    obs_sub = adata.obs.iloc[idx]

    logger.info(f"Running UMAP on {n} cells...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    embedding = reducer.fit_transform(latent_sub)

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Extract donor_age
    import re

    def extract_age(stage):
        match = re.search(r"(\d+)-year", str(stage))
        return int(match.group(1)) if match else None

    if "donor_age" not in obs_sub.columns:
        obs_sub = obs_sub.copy()
        obs_sub["donor_age"] = obs_sub["development_stage"].apply(extract_age)

    # Plot 1: Colored by age
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=obs_sub["donor_age"].values,
                    cmap="RdYlBu_r", s=1, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Donor Age")
    ax.set_title("Latent Space UMAP — Colored by Age")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.savefig(fig_dir / "umap_age.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved {fig_dir / 'umap_age.png'}")

    # Plot 2: Colored by cell type (top 15)
    top_types = obs_sub["cell_type"].value_counts().head(15).index
    mask = obs_sub["cell_type"].isin(top_types)
    fig, ax = plt.subplots(figsize=(12, 8))
    for ct in top_types:
        ct_mask = obs_sub["cell_type"] == ct
        ax.scatter(embedding[ct_mask, 0], embedding[ct_mask, 1], s=1, alpha=0.5, label=ct)
    ax.legend(markerscale=5, fontsize=7, loc="best")
    ax.set_title("Latent Space UMAP — Colored by Cell Type (top 15)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.savefig(fig_dir / "umap_celltype.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved {fig_dir / 'umap_celltype.png'}")

    # Plot 3: Colored by sex
    fig, ax = plt.subplots(figsize=(10, 8))
    for sex in obs_sub["sex"].unique():
        sex_mask = obs_sub["sex"] == sex
        ax.scatter(embedding[sex_mask, 0], embedding[sex_mask, 1], s=1, alpha=0.5, label=sex)
    ax.legend(markerscale=5)
    ax.set_title("Latent Space UMAP — Colored by Sex")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.savefig(fig_dir / "umap_sex.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved {fig_dir / 'umap_sex.png'}")

    print(f"\nFigures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
