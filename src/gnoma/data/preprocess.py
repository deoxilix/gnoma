"""Preprocessing pipeline for single-cell aging data (T-004, T-005).

Implements reproducible QC, normalization, HVG selection,
and donor-stratified train/val/test splitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline."""

    # QC thresholds
    min_genes_per_cell: int = 200
    max_genes_per_cell: int = 8000
    max_pct_mito: float = 20.0
    min_cells_per_gene: int = 10

    # Normalization
    target_sum: float = 1e4

    # HVG selection
    n_top_genes: int = 4000
    hvg_flavor: str = "seurat_v3"  # requires raw counts

    # Dimensionality reduction (for validation / visualization)
    n_pcs: int = 50
    n_neighbors: int = 15

    # Split
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 42


def run_qc(adata: ad.AnnData, config: PreprocessConfig) -> ad.AnnData:
    """Run quality control filtering.

    Filters cells by gene count and mitochondrial fraction.
    Filters genes by minimum cell count.
    """
    logger.info(f"Pre-QC: {adata.n_obs} cells, {adata.n_vars} genes")

    # Compute QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)

    # Filter cells
    cell_mask = (
        (adata.obs["n_genes_by_counts"] >= config.min_genes_per_cell)
        & (adata.obs["n_genes_by_counts"] <= config.max_genes_per_cell)
        & (adata.obs["pct_counts_mt"] <= config.max_pct_mito)
    )
    adata = adata[cell_mask].copy()

    # Filter genes
    sc.pp.filter_genes(adata, min_cells=config.min_cells_per_gene)

    logger.info(f"Post-QC: {adata.n_obs} cells, {adata.n_vars} genes")
    return adata


def normalize(adata: ad.AnnData, config: PreprocessConfig) -> ad.AnnData:
    """Normalize and log-transform, preserving raw counts."""
    # Store raw counts
    adata.raw = adata.copy()

    sc.pp.normalize_total(adata, target_sum=config.target_sum)
    sc.pp.log1p(adata)

    return adata


def select_hvg(adata: ad.AnnData, config: PreprocessConfig) -> ad.AnnData:
    """Select highly variable genes."""
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=config.n_top_genes,
        flavor=config.hvg_flavor,
        subset=False,
    )
    logger.info(f"Selected {adata.var['highly_variable'].sum()} highly variable genes")
    return adata


def compute_embeddings(adata: ad.AnnData, config: PreprocessConfig) -> ad.AnnData:
    """Compute PCA and UMAP for visualization and validation."""
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=config.n_pcs)
    sc.pp.neighbors(adata, n_neighbors=config.n_neighbors, n_pcs=config.n_pcs)
    sc.tl.umap(adata)
    return adata


def split_by_donor(
    adata: ad.AnnData,
    config: PreprocessConfig,
    donor_col: str = "donor_id",
) -> dict[str, ad.AnnData]:
    """Split data by donor ID to prevent leakage.

    Stratifies by age group if 'age_group' column exists,
    otherwise random donor split.

    Returns:
        Dict with keys 'train', 'val', 'test'.
    """
    rng = np.random.RandomState(config.split_seed)
    donors = adata.obs[donor_col].unique()
    rng.shuffle(donors)

    n_val = max(1, int(len(donors) * config.val_fraction))
    n_test = max(1, int(len(donors) * config.test_fraction))

    test_donors = set(donors[:n_test])
    val_donors = set(donors[n_test : n_test + n_val])
    train_donors = set(donors[n_test + n_val :])

    splits = {
        "train": adata[adata.obs[donor_col].isin(train_donors)].copy(),
        "val": adata[adata.obs[donor_col].isin(val_donors)].copy(),
        "test": adata[adata.obs[donor_col].isin(test_donors)].copy(),
    }

    for name, split in splits.items():
        n_donors = split.obs[donor_col].nunique()
        logger.info(f"Split '{name}': {split.n_obs} cells, {n_donors} donors")

    return splits


def run_pipeline(
    adata: ad.AnnData,
    config: Optional[PreprocessConfig] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, ad.AnnData]:
    """Run the full preprocessing pipeline.

    Args:
        adata: Raw AnnData.
        config: Preprocessing config. Uses defaults if None.
        output_dir: If provided, save processed splits to this directory.

    Returns:
        Dict with 'train', 'val', 'test' AnnData splits.
    """
    if config is None:
        config = PreprocessConfig()

    logger.info("Starting preprocessing pipeline")

    adata = run_qc(adata, config)
    adata = normalize(adata, config)
    adata = select_hvg(adata, config)

    splits = split_by_donor(adata, config)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, split in splits.items():
            path = output_dir / f"{name}.h5ad"
            split.write_h5ad(path)
            logger.info(f"Saved {name} split to {path}")

    return splits
