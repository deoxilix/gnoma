#!/usr/bin/env python3
"""Download a focused blood/PBMC subset for MVP development (T-003).

Downloads age-labeled normal blood cells from cellxgene-census,
targeting the recommended 10x 5' v2 dataset with broad age coverage.

Outputs: data/raw/blood_aging_subset.h5ad
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# Target: ~50k cells across a wide age range for development
MAX_CELLS = 50_000
MIN_AGE = 20
MAX_AGE = 80


def main():
    try:
        import cellxgene_census
        import numpy as np
    except ImportError:
        logger.error("Missing dependencies")
        sys.exit(1)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "blood_aging_subset.h5ad"

    if output_path.exists():
        logger.info(f"{output_path} already exists, skipping download")
        return

    logger.info("Opening census (stable)...")
    with cellxgene_census.open_soma(census_version="2025-11-08") as census:
        # Simple filter — get all normal blood metadata, filter age locally
        value_filter = "tissue_general == 'blood' and disease == 'normal'"

        logger.info("Querying blood cell metadata...")
        obs_df = cellxgene_census.get_obs(
            census,
            organism="Homo sapiens",
            value_filter=value_filter,
            column_names=[
                "soma_joinid",
                "dataset_id",
                "donor_id",
                "cell_type",
                "tissue_general",
                "development_stage",
                "sex",
                "assay",
            ],
        )

        logger.info(f"Total normal blood cells: {len(obs_df):,}")

        # Extract age and filter locally
        import re

        def extract_age(stage):
            match = re.search(r"(\d+)-year", str(stage))
            return int(match.group(1)) if match else None

        obs_df["donor_age"] = obs_df["development_stage"].apply(extract_age)
        obs_df = obs_df[obs_df["donor_age"].between(MIN_AGE, MAX_AGE)]
        logger.info(f"After age filter ({MIN_AGE}–{MAX_AGE}): {len(obs_df):,} cells")

        logger.info(f"Total matching cells: {len(obs_df):,}")
        logger.info(f"Unique donors: {obs_df['donor_id'].nunique()}")

        if len(obs_df) == 0:
            logger.error("No cells matched the filter. Check census version and filters.")
            sys.exit(1)

        # Subsample: stratify by donor to preserve age diversity
        if len(obs_df) > MAX_CELLS:
            donors = obs_df["donor_id"].unique()
            cells_per_donor = max(1, MAX_CELLS // len(donors))
            logger.info(
                f"Subsampling to ~{MAX_CELLS} cells "
                f"({cells_per_donor} per donor, {len(donors)} donors)"
            )
            sampled_indices = []
            rng = np.random.RandomState(42)
            for donor in donors:
                donor_cells = obs_df[obs_df["donor_id"] == donor]
                n = min(len(donor_cells), cells_per_donor)
                idx = rng.choice(donor_cells.index, n, replace=False)
                sampled_indices.extend(idx)

            # If we didn't reach MAX_CELLS, sample more from large donors
            if len(sampled_indices) < MAX_CELLS:
                remaining = MAX_CELLS - len(sampled_indices)
                available = obs_df.index.difference(sampled_indices)
                extra = rng.choice(available, min(remaining, len(available)), replace=False)
                sampled_indices.extend(extra)

            obs_subset = obs_df.loc[sampled_indices]
        else:
            obs_subset = obs_df

        logger.info(f"Selected {len(obs_subset):,} cells from {obs_subset['donor_id'].nunique()} donors")

        # Get soma_joinids for the subset
        soma_ids = obs_subset["soma_joinid"].tolist()

        logger.info("Downloading expression data (this may take a few minutes)...")
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_coords=soma_ids,
            obs_column_names=[
                "dataset_id",
                "donor_id",
                "cell_type",
                "tissue_general",
                "development_stage",
                "sex",
                "assay",
            ],
        )

        logger.info(f"Downloaded: {adata.n_obs} cells x {adata.n_vars} genes")

        # Extract numeric age
        import re

        def extract_age(stage):
            match = re.search(r"(\d+)-year", str(stage))
            return int(match.group(1)) if match else None

        adata.obs["donor_age"] = adata.obs["development_stage"].apply(extract_age)

        # Summary
        print(f"\n=== Blood Aging Subset ===")
        print(f"Cells: {adata.n_obs:,}")
        print(f"Genes: {adata.n_vars:,}")
        print(f"Donors: {adata.obs['donor_id'].nunique()}")
        print(f"Age range: {adata.obs['donor_age'].min()}–{adata.obs['donor_age'].max()}")
        print(f"Cell types: {adata.obs['cell_type'].nunique()}")
        print(f"\nAge distribution:")
        age_bins = adata.obs["donor_age"].value_counts().sort_index()
        for age, count in age_bins.head(20).items():
            print(f"  {age}: {count}")

        # Save
        adata.write_h5ad(output_path)
        logger.info(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
