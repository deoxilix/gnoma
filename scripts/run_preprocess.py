#!/usr/bin/env python3
"""Run preprocessing pipeline on downloaded blood subset (T-004, T-005).

Reads data/raw/blood_aging_subset.h5ad and produces train/val/test splits
in data/processed/.
"""

import logging
import sys
from pathlib import Path

import anndata as ad

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    raw_path = Path("data/raw/blood_aging_subset.h5ad")
    output_dir = Path("data/processed")

    if not raw_path.exists():
        logger.error(f"{raw_path} not found. Run download_blood_subset.py first.")
        sys.exit(1)

    logger.info(f"Loading {raw_path}...")
    adata = ad.read_h5ad(raw_path)
    logger.info(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # Add donor_age if not present
    if "donor_age" not in adata.obs.columns:
        import re

        def extract_age(stage):
            match = re.search(r"(\d+)-year", str(stage))
            return int(match.group(1)) if match else None

        adata.obs["donor_age"] = adata.obs["development_stage"].apply(extract_age)

    # Summary before preprocessing
    print(f"\n=== Raw Data Summary ===")
    print(f"Cells: {adata.n_obs:,}")
    print(f"Genes: {adata.n_vars:,}")
    print(f"Donors: {adata.obs['donor_id'].nunique()}")
    if "donor_age" in adata.obs.columns:
        ages = adata.obs["donor_age"].dropna()
        print(f"Age range: {ages.min()}-{ages.max()}")
    print(f"Cell types: {adata.obs['cell_type'].nunique()}")

    # Run pipeline
    from gnoma.data.preprocess import PreprocessConfig, run_pipeline

    config = PreprocessConfig()
    splits = run_pipeline(adata, config=config, output_dir=output_dir)

    # Summary after preprocessing
    print(f"\n=== Processed Data Summary ===")
    for name, split in splits.items():
        n_donors = split.obs["donor_id"].nunique()
        n_hvg = split.var["highly_variable"].sum() if "highly_variable" in split.var.columns else "N/A"
        print(f"  {name}: {split.n_obs:,} cells, {n_donors} donors, {n_hvg} HVGs")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
