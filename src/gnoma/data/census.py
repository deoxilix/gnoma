"""Census data access utilities (T-001, T-002).

Provides functions to query and download single-cell data from
the CZI cellxgene-census for aging-relevant studies.
"""

from __future__ import annotations

import logging
from typing import Optional

import anndata as ad
import cellxgene_census
import pandas as pd

logger = logging.getLogger(__name__)


def open_census(census_version: str = "stable"):
    """Open a cellxgene-census connection.

    Returns a context-managed census object. Use with `with` statement.
    """
    return cellxgene_census.open_soma(census_version=census_version)


def query_aging_metadata(
    census,
    organism: str = "Homo sapiens",
    tissue: Optional[str] = None,
) -> pd.DataFrame:
    """Query census metadata for entries with donor age annotations.

    Args:
        census: Open census connection.
        organism: Species filter.
        tissue: Optional tissue filter (e.g., 'blood', 'skin').

    Returns:
        DataFrame of dataset/donor metadata with age information.
    """
    obs_df = cellxgene_census.get_obs(
        census,
        organism=organism,
        column_names=[
            "dataset_id",
            "donor_id",
            "cell_type",
            "tissue_general",
            "development_stage",
            "disease",
            "sex",
            "assay",
        ],
    )

    if tissue is not None:
        obs_df = obs_df[obs_df["tissue_general"].str.contains(tissue, case=False, na=False)]

    summary = (
        obs_df.groupby(["dataset_id", "tissue_general", "assay"])
        .agg(
            n_cells=("donor_id", "count"),
            n_donors=("donor_id", "nunique"),
            cell_types=("cell_type", lambda x: list(x.unique()[:10])),
        )
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )

    return summary


def fetch_subset(
    census,
    organism: str = "Homo sapiens",
    tissue: str = "blood",
    cell_type: Optional[str] = None,
    max_cells: int = 5000,
    obs_value_filter: Optional[str] = None,
) -> ad.AnnData:
    """Fetch a small subset of cells from census for initial exploration.

    Args:
        census: Open census connection.
        organism: Species.
        tissue: Tissue filter.
        cell_type: Optional cell type filter.
        max_cells: Maximum cells to return.
        obs_value_filter: Raw SOMA value_filter string for obs.

    Returns:
        AnnData with raw counts and obs metadata.
    """
    filters = []
    if tissue:
        filters.append(f"tissue_general == '{tissue}'")
    if cell_type:
        filters.append(f"cell_type == '{cell_type}'")
    if obs_value_filter:
        filters.append(obs_value_filter)

    value_filter = " and ".join(filters) if filters else None

    logger.info(f"Fetching census subset: organism={organism}, filter={value_filter}")

    adata = cellxgene_census.get_anndata(
        census,
        organism=organism,
        obs_value_filter=value_filter,
    )

    if adata.n_obs > max_cells:
        import numpy as np

        idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata = adata[sorted(idx)].copy()
        logger.info(f"Subsampled to {max_cells} cells")

    logger.info(f"Fetched AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata
