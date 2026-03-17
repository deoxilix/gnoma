#!/usr/bin/env python3
"""Inventory aging-relevant datasets in cellxgene-census (T-002, T-003).

Run: python scripts/dataset_inventory.py

Produces:
- data/dataset_inventory.csv with study-level summaries
- Console output with MVP cell type recommendation
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# Tissues of interest for aging studies
TARGET_TISSUES = ["blood", "skin", "lung", "brain", "liver", "kidney", "heart"]

# Minimum thresholds for a usable dataset
MIN_DONORS = 10
MIN_AGE_RANGE_YEARS = 30


def main():
    try:
        import cellxgene_census
    except ImportError:
        logger.error("cellxgene-census not installed")
        sys.exit(1)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    logger.info("Opening census connection...")
    with cellxgene_census.open_soma(census_version="stable") as census:
        logger.info("Querying human cell metadata (this may take a few minutes)...")

        # Get obs metadata for normal (non-disease) human cells
        obs_df = cellxgene_census.get_obs(
            census,
            organism="Homo sapiens",
            value_filter="disease == 'normal'",
            column_names=[
                "dataset_id",
                "donor_id",
                "cell_type",
                "tissue_general",
                "development_stage",
                "sex",
                "assay",
            ],
        )

        logger.info(f"Total normal human cells: {len(obs_df):,}")

        # Filter to target tissues
        tissue_mask = obs_df["tissue_general"].str.lower().isin(TARGET_TISSUES)
        obs_filtered = obs_df[tissue_mask].copy()
        logger.info(f"Cells in target tissues: {len(obs_filtered):,}")

        # Extract age from development_stage where possible
        # development_stage contains entries like "45-year-old human stage"
        def extract_age(stage):
            if pd.isna(stage):
                return None
            s = str(stage).lower()
            for token in s.split("-"):
                if token.strip().endswith("year"):
                    try:
                        return int(token.strip().replace("year", "").strip())
                    except ValueError:
                        pass
            # Try pattern like "45-year-old"
            import re

            match = re.search(r"(\d+)-year", s)
            if match:
                return int(match.group(1))
            return None

        obs_filtered["donor_age"] = obs_filtered["development_stage"].apply(extract_age)
        has_age = obs_filtered["donor_age"].notna()
        logger.info(f"Cells with parseable age: {has_age.sum():,} ({has_age.mean()*100:.1f}%)")

        obs_aged = obs_filtered[has_age].copy()

        # Study-level summary
        study_summary = (
            obs_aged.groupby(["dataset_id", "tissue_general", "assay"])
            .agg(
                n_cells=("donor_id", "count"),
                n_donors=("donor_id", "nunique"),
                age_min=("donor_age", "min"),
                age_max=("donor_age", "max"),
                age_median=("donor_age", "median"),
                n_cell_types=("cell_type", "nunique"),
                top_cell_types=("cell_type", lambda x: "; ".join(x.value_counts().head(5).index)),
                sexes=("sex", lambda x: "; ".join(sorted(x.unique()))),
            )
            .reset_index()
        )
        study_summary["age_range"] = study_summary["age_max"] - study_summary["age_min"]
        study_summary = study_summary.sort_values("n_cells", ascending=False)

        # Save full inventory
        inventory_path = output_dir / "dataset_inventory.csv"
        study_summary.to_csv(inventory_path, index=False)
        logger.info(f"Saved inventory to {inventory_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("DATASET INVENTORY — Aging-Relevant Studies in cellxgene-census")
        print("=" * 80)

        # Per-tissue summary
        print("\n--- By Tissue ---")
        tissue_agg = (
            obs_aged.groupby("tissue_general")
            .agg(
                n_cells=("donor_id", "count"),
                n_donors=("donor_id", "nunique"),
                age_min=("donor_age", "min"),
                age_max=("donor_age", "max"),
                n_datasets=("dataset_id", "nunique"),
            )
            .sort_values("n_cells", ascending=False)
        )
        print(tissue_agg.to_string())

        # MVP cell type recommendation
        print("\n--- MVP Cell Type Recommendation ---")
        usable = study_summary[
            (study_summary["n_donors"] >= MIN_DONORS)
            & (study_summary["age_range"] >= MIN_AGE_RANGE_YEARS)
        ].sort_values("n_cells", ascending=False)

        if len(usable) > 0:
            best = usable.iloc[0]
            print(f"\nRecommended: {best['tissue_general']} ({best['assay']})")
            print(f"  Dataset: {best['dataset_id']}")
            print(f"  Cells: {best['n_cells']:,}")
            print(f"  Donors: {best['n_donors']}")
            print(f"  Age range: {best['age_min']:.0f}–{best['age_max']:.0f}")
            print(f"  Cell types: {best['n_cell_types']}")
            print(f"  Top types: {best['top_cell_types']}")
        else:
            print("No datasets met minimum criteria. Consider relaxing thresholds.")

        # Top 10 datasets
        print("\n--- Top 10 Datasets by Cell Count ---")
        for _, row in usable.head(10).iterrows():
            print(
                f"  {row['tissue_general']:>8s} | {row['n_cells']:>8,} cells | "
                f"{row['n_donors']:>3} donors | ages {row['age_min']:.0f}–{row['age_max']:.0f} | "
                f"{row['assay']}"
            )

        print(f"\nTotal usable datasets: {len(usable)}")
        print(f"Total inventory saved: {inventory_path}")


if __name__ == "__main__":
    main()
