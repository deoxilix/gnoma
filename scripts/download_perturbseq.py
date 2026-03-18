#!/usr/bin/env python3
"""Download and preprocess Replogle 2022 K562 essential Perturb-seq data (Sprint 6).

Downloads the scPerturb-harmonized K562 essential CRISPRi Perturb-seq dataset
and preprocesses it for training the neural transition model. Maps perturbation
targets to our intervention ontology.

Source: Replogle et al., Cell, 2022
"Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq"
Harmonized by: scPerturb (Peidli et al., Nat Methods, 2024)

Downloads from: https://zenodo.org/records/13350497
"""

import json
import logging
import sys
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# scPerturb harmonized version on Zenodo (1.5 GB)
ZENODO_URL = "https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1"


def download_with_progress(url: str, dest: Path) -> None:
    """Download file with progress reporting."""
    logger.info(f"Downloading from Zenodo (scPerturb harmonized)...")
    logger.info(f"  → {dest}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            mb = count * block_size / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {pct:.1f}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)
        else:
            mb = count * block_size / 1e6
            print(f"\r  {mb:.0f} MB downloaded", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=reporthook)
    print()
    size_mb = dest.stat().st_size / 1e6
    logger.info(f"Download complete: {size_mb:.0f} MB")
    if size_mb < 1:
        logger.error("File too small — download likely failed")
        sys.exit(1)


def main():
    output_dir = Path("data/perturbseq")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "ReplogleWeissman2022_K562_essential.h5ad"

    # --- Download if needed ---
    if raw_path.exists() and raw_path.stat().st_size > 1e6:
        logger.info(f"Data already exists: {raw_path} ({raw_path.stat().st_size / 1e6:.0f} MB)")
    else:
        download_with_progress(ZENODO_URL, raw_path)

    # --- Load and inspect ---
    logger.info("Loading Perturb-seq data...")
    adata = ad.read_h5ad(raw_path)
    logger.info(f"Shape: {adata.shape}")
    logger.info(f"obs columns: {list(adata.obs.columns)}")
    logger.info(f"var columns: {list(adata.var.columns)}")

    # scPerturb standardized columns: 'perturbation', 'is_control', 'gene_target', etc.
    perturbation_col = None
    for col in ["perturbation", "gene", "perturbation_name", "target_gene", "gene_target"]:
        if col in adata.obs.columns:
            perturbation_col = col
            break

    if perturbation_col is None:
        logger.info("obs columns and sample values:")
        for col in adata.obs.columns:
            logger.info(f"  {col}: {adata.obs[col].nunique()} unique, sample: {adata.obs[col].iloc[:3].tolist()}")
        logger.error("Could not identify perturbation column.")
        sys.exit(1)

    logger.info(f"Perturbation column: '{perturbation_col}'")
    n_perturbations = adata.obs[perturbation_col].nunique()
    logger.info(f"Unique perturbations: {n_perturbations}")

    # Top perturbations by cell count
    perts = adata.obs[perturbation_col].value_counts()
    logger.info(f"Top 10 perturbations by cell count:")
    for p, c in perts.head(10).items():
        logger.info(f"  {p}: {c} cells")

    # Identify control cells (scPerturb uses 'control' column or 'is_control')
    if "is_control" in adata.obs.columns:
        control_mask = adata.obs["is_control"].astype(bool)
    else:
        control_labels = {"non-targeting", "control", "NTC", "non_targeting", "NT", "CTRL"}
        control_mask = adata.obs[perturbation_col].isin(control_labels)
        if control_mask.sum() == 0:
            lower_vals = adata.obs[perturbation_col].str.lower()
            for label in control_labels:
                control_mask |= lower_vals == label.lower()

    n_control = int(control_mask.sum())
    logger.info(f"Control cells: {n_control}")

    # Extract gene target names from perturbation labels
    # scPerturb format is typically "GENE_NAME" or "GENE_NAME_guide1"
    perturb_values = set(adata.obs.loc[~control_mask, perturbation_col].unique())
    # Extract base gene name (before any suffix like _guide, _1, etc.)
    perturb_genes = set()
    for p in perturb_values:
        gene = str(p).split("_")[0] if "_" in str(p) else str(p)
        perturb_genes.add(gene)

    logger.info(f"Unique perturbation gene targets: {len(perturb_genes)}")

    # --- Map to our intervention ontology ---
    ontology = json.loads(Path("data/interventions/ontology_v1.json").read_text())
    our_genes = {
        i["target_gene"]: i["id"]
        for i in ontology["interventions"]
        if i.get("target_gene")
    }
    logger.info(f"Our ontology has {len(our_genes)} gene targets")

    matched = perturb_genes & set(our_genes.keys())
    logger.info(f"Matched to our ontology: {len(matched)}")
    if matched:
        logger.info(f"Matched genes: {sorted(matched)}")

    # Also check raw perturbation values for matches
    raw_matched = set()
    for p in perturb_values:
        p_str = str(p)
        if p_str in our_genes:
            raw_matched.add(p_str)
        # Also check if gene name appears as prefix
        gene = p_str.split("_")[0] if "_" in p_str else p_str
        if gene in our_genes:
            raw_matched.add(p_str)

    logger.info(f"Raw perturbation labels matching ontology: {len(raw_matched)}")

    # --- Load our HVG list ---
    train_path = Path("data/processed/train.h5ad")
    hvg_overlap = 0
    if train_path.exists():
        adata_train = ad.read_h5ad(train_path)
        if "feature_name" in adata_train.var.columns:
            our_genes_list = list(adata_train.var["feature_name"])
        else:
            our_genes_list = list(adata_train.var_names)

        hvg_mask = adata_train.var.get("highly_variable")
        if hvg_mask is not None:
            col = "feature_name" if "feature_name" in adata_train.var.columns else None
            our_hvgs = set(adata_train.var[col][hvg_mask]) if col else set(adata_train.var_names[hvg_mask])
        else:
            our_hvgs = set(our_genes_list)
        logger.info(f"Our HVGs: {len(our_hvgs)}")
        del adata_train

        # Gene names in perturb-seq data
        ps_gene_col = None
        for col in ["gene_name", "feature_name", "gene_symbol"]:
            if col in adata.var.columns:
                ps_gene_col = col
                break
        if ps_gene_col:
            ps_genes = set(adata.var[ps_gene_col])
        else:
            ps_genes = set(adata.var_names)

        hvg_overlap = len(our_hvgs & ps_genes)
        logger.info(f"Perturb-seq gene name column: {ps_gene_col or 'var_names'}")
        logger.info(f"Perturb-seq genes: {len(ps_genes)}")
        logger.info(f"HVG overlap: {hvg_overlap} / {len(our_hvgs)}")

    # --- Save preprocessed subset ---
    # Keep control + all perturbation cells (we need all for training)
    # But filter to perturbations with ≥10 cells for statistical power
    min_cells = 10
    good_perts = set(perts[perts >= min_cells].index) - ({"control", "non-targeting"} if "is_control" not in adata.obs.columns else set())

    # Save matched subset for transition model training
    if raw_matched:
        keep_perts = raw_matched | {p for p in perts.index if control_mask[adata.obs[perturbation_col] == p].any()}
        keep_mask = control_mask.copy()
        for p in raw_matched:
            keep_mask |= adata.obs[perturbation_col] == p

        adata_sub = adata[keep_mask].copy()
        logger.info(f"Matched subset shape: {adata_sub.shape}")

        out_path = output_dir / "K562_essential_matched.h5ad"
        adata_sub.write_h5ad(out_path)
        logger.info(f"Saved matched subset to {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    else:
        logger.warning("No direct gene matches found.")
        # Save the full dataset anyway — we'll use all perturbations for training
        logger.info("Will use all perturbations for general transition model training.")

    # --- Save mapping info ---
    mapping = {
        "source": "Replogle et al., Cell, 2022 (scPerturb harmonized)",
        "dataset": "K562_essential",
        "zenodo_url": ZENODO_URL.split("?")[0],
        "perturbation_column": perturbation_col,
        "n_cells_total": int(adata.shape[0]),
        "n_genes_total": int(adata.shape[1]),
        "n_control_cells": n_control,
        "n_perturbation_targets": int(len(perturb_genes)),
        "n_matched_to_ontology": int(len(matched)),
        "matched_genes": sorted(matched),
        "hvg_overlap": hvg_overlap,
        "min_cells_per_perturbation": min_cells,
        "n_perturbations_with_min_cells": int(len(good_perts)),
    }
    mapping_path = output_dir / "perturbseq_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved mapping to {mapping_path}")

    print(f"\n=== Perturb-seq Download Summary ===")
    print(f"Dataset: Replogle 2022 K562 Essential CRISPRi (scPerturb)")
    print(f"Total cells: {adata.shape[0]:,}")
    print(f"Total genes: {adata.shape[1]:,}")
    print(f"Control cells: {n_control:,}")
    print(f"Perturbation targets: {len(perturb_genes):,}")
    print(f"Matched to ontology: {len(matched)}")
    print(f"HVG overlap: {hvg_overlap}")
    if matched:
        print(f"Matched genes: {', '.join(sorted(matched))}")


if __name__ == "__main__":
    main()
