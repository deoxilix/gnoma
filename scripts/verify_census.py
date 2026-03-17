#!/usr/bin/env python3
"""Verify cellxgene-census API access (T-001).

Run: python scripts/verify_census.py

Expected output: summary of a small cell subset with age annotations.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    try:
        import cellxgene_census
    except ImportError:
        logger.error("cellxgene-census not installed. Run: pip install cellxgene-census")
        sys.exit(1)

    logger.info("Opening census connection (stable version)...")
    with cellxgene_census.open_soma(census_version="stable") as census:
        # Check census version info
        logger.info(f"Census object type: {type(census)}")

        # Query a small subset of human blood cells
        logger.info("Fetching small PBMC subset...")
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter="tissue_general == 'blood' and disease == 'normal'",
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

        logger.info(f"Total cells returned: {adata.n_obs}")
        logger.info(f"Total genes: {adata.n_vars}")
        logger.info(f"Columns: {list(adata.obs.columns)}")

        # Summarize
        print("\n=== Census Access Verified ===")
        print(f"Cells: {adata.n_obs:,}")
        print(f"Genes: {adata.n_vars:,}")
        print(f"\nUnique donors: {adata.obs['donor_id'].nunique()}")
        print(f"Unique cell types: {adata.obs['cell_type'].nunique()}")
        print(f"\nTop cell types:")
        print(adata.obs["cell_type"].value_counts().head(10).to_string())
        print(f"\nDevelopment stages (age proxy):")
        print(adata.obs["development_stage"].value_counts().head(10).to_string())
        print(f"\nAssays:")
        print(adata.obs["assay"].value_counts().to_string())

    logger.info("Census access verified successfully.")


if __name__ == "__main__":
    main()
