"""Tests for preprocessing pipeline (T-004)."""

import anndata as ad
import numpy as np

from gnoma.data.preprocess import PreprocessConfig, normalize, run_qc, split_by_donor


def _make_synthetic_adata(n_cells=500, n_genes=2000, n_donors=10):
    """Create synthetic AnnData for testing."""
    rng = np.random.RandomState(42)

    # Sparse count matrix
    counts = rng.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Make some genes mitochondrial
    var_names = [f"GENE{i}" for i in range(n_genes)]
    var_names[0] = "MT-CO1"
    var_names[1] = "MT-CO2"
    var_names[2] = "MT-ND1"

    # Donor assignment
    donors = [f"donor_{i % n_donors}" for i in range(n_cells)]
    ages = [30 + (i % n_donors) * 5 for i in range(n_cells)]

    adata = ad.AnnData(
        X=counts,
        var={"gene_name": var_names},
    )
    adata.var_names = var_names
    adata.obs["donor_id"] = donors
    adata.obs["donor_age"] = ages

    return adata


def test_qc_filters_cells():
    adata = _make_synthetic_adata()
    config = PreprocessConfig(min_genes_per_cell=10, max_pct_mito=50.0)
    filtered = run_qc(adata, config)
    assert filtered.n_obs <= adata.n_obs
    assert filtered.n_obs > 0


def test_normalize_preserves_raw():
    adata = _make_synthetic_adata()
    config = PreprocessConfig(min_genes_per_cell=10, max_pct_mito=50.0)
    adata = run_qc(adata, config)
    normalized = normalize(adata, config)
    assert normalized.raw is not None
    assert normalized.raw.X.shape == normalized.X.shape


def test_split_by_donor_no_leakage():
    adata = _make_synthetic_adata(n_cells=500, n_donors=10)
    config = PreprocessConfig()
    splits = split_by_donor(adata, config)

    train_donors = set(splits["train"].obs["donor_id"])
    val_donors = set(splits["val"].obs["donor_id"])
    test_donors = set(splits["test"].obs["donor_id"])

    # No overlap between splits
    assert len(train_donors & val_donors) == 0
    assert len(train_donors & test_donors) == 0
    assert len(val_donors & test_donors) == 0

    # All donors accounted for
    all_donors = train_donors | val_donors | test_donors
    assert all_donors == set(adata.obs["donor_id"].unique())
