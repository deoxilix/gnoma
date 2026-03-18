#!/usr/bin/env python3
"""Train scVI encoder and aging clock on preprocessed blood data (T-010, T-011, T-012).

Expects data/processed/{train,val,test}.h5ad from run_preprocess.py.
Outputs models to models/encoder/ and models/aging_clock/.
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
    model_dir = Path("models")

    # Load splits
    train_path = processed_dir / "train.h5ad"
    val_path = processed_dir / "val.h5ad"

    if not train_path.exists() or not val_path.exists():
        logger.error("Processed data not found. Run scripts/run_preprocess.py first.")
        sys.exit(1)

    logger.info("Loading preprocessed data...")
    adata_train = ad.read_h5ad(train_path)
    adata_val = ad.read_h5ad(val_path)
    logger.info(f"Train: {adata_train.n_obs} cells, Val: {adata_val.n_obs} cells")

    # Prepare HVG subset for scVI (which operates on HVG-only data)
    if "highly_variable" in adata_train.var.columns:
        hvg_mask = adata_train.var["highly_variable"]
        adata_train_hvg = adata_train[:, hvg_mask].copy()
        adata_val_hvg = adata_val[:, hvg_mask].copy()
    else:
        adata_train_hvg = adata_train
        adata_val_hvg = adata_val
    logger.info(f"HVG subset: {adata_train_hvg.n_vars} genes")

    # --- Step 1: Train scVI encoder ---
    # Note: batch_key=None to avoid donor registry issues when encoding val/test splits
    encoder_dir = model_dir / "encoder"
    if encoder_dir.exists() and (encoder_dir / "model.pt").exists():
        logger.info("Encoder already trained, loading...")
        import scvi

        scvi.model.SCVI.setup_anndata(adata_train_hvg)
        encoder_model = scvi.model.SCVI.load(str(encoder_dir), adata=adata_train_hvg)
    else:
        from gnoma.models.encoder import BiologicalEncoder, EncoderConfig

        config = EncoderConfig(
            latent_dim=128,
            n_layers=2,
            n_hidden=256,
            max_epochs=200,
            batch_key=None,  # Skip batch correction for MVP — avoids donor registry issues
        )
        encoder = BiologicalEncoder(config)
        train_raw = adata_train_hvg

        encoder.setup(train_raw)
        encoder.train()
        encoder.save(encoder_dir)
        encoder_model = encoder.model

    # --- Step 2: Encode all cells ---
    logger.info("Encoding cells to latent space...")
    latent_train = encoder_model.get_latent_representation(adata_train_hvg)
    latent_val = encoder_model.get_latent_representation(adata_val_hvg)

    logger.info(f"Latent shapes: train={latent_train.shape}, val={latent_val.shape}")

    # Save latent embeddings
    latent_dir = Path("data/latent")
    latent_dir.mkdir(parents=True, exist_ok=True)
    np.save(latent_dir / "latent_train.npy", latent_train)
    np.save(latent_dir / "latent_val.npy", latent_val)
    logger.info(f"Saved latent embeddings to {latent_dir}")

    # --- Step 3: Train aging clock ---
    logger.info("Training aging clock...")

    # Extract ages
    import re

    def extract_age(stage):
        match = re.search(r"(\d+)-year", str(stage))
        return int(match.group(1)) if match else None

    if "donor_age" not in adata_train.obs.columns:
        adata_train.obs["donor_age"] = adata_train.obs["development_stage"].apply(extract_age)
    if "donor_age" not in adata_val.obs.columns:
        adata_val.obs["donor_age"] = adata_val.obs["development_stage"].apply(extract_age)

    ages_train = adata_train.obs["donor_age"].values.astype(np.float32)
    ages_val = adata_val.obs["donor_age"].values.astype(np.float32)

    # Filter out NaN ages
    valid_train = ~np.isnan(ages_train)
    valid_val = ~np.isnan(ages_val)

    from gnoma.models.aging_clock import AgingClockConfig, train_aging_clock

    clock_config = AgingClockConfig(hidden_dim=64, max_epochs=100, patience=15)
    clock_model, clock_metrics = train_aging_clock(
        latent_train[valid_train],
        ages_train[valid_train],
        latent_val[valid_val],
        ages_val[valid_val],
        config=clock_config,
    )

    # Save aging clock
    clock_dir = model_dir / "aging_clock"
    clock_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "state_dict": clock_model.state_dict(),
            "config": clock_config,
            "age_mean": clock_model._age_mean,
            "age_std": clock_model._age_std,
            "metrics": {k: v for k, v in clock_metrics.items() if k not in ("train_losses", "val_losses")},
        },
        clock_dir / "aging_clock.pt",
    )

    # Save metrics
    metrics_out = {k: v for k, v in clock_metrics.items() if k not in ("train_losses", "val_losses")}
    with open(clock_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\n=== Training Summary ===")
    print(f"Encoder: {latent_train.shape[1]}-dim latent, saved to {encoder_dir}")
    print(f"Aging clock: R²={clock_metrics['r2']:.3f}, MAE={clock_metrics['mae']:.1f} years")
    print(f"  Saved to {clock_dir}")


if __name__ == "__main__":
    main()
