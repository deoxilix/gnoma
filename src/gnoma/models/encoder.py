"""Biological state encoder using scVI (T-010, T-011).

Wraps scvi-tools to train a VAE on single-cell aging data and
extract compressed latent representations for downstream RL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """Configuration for scVI encoder training."""

    latent_dim: int = 128
    n_layers: int = 2
    n_hidden: int = 256
    dropout_rate: float = 0.1
    gene_likelihood: str = "zinb"  # zero-inflated negative binomial
    max_epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 1e-3
    early_stopping: bool = True
    early_stopping_patience: int = 10
    batch_key: Optional[str] = "donor_id"  # batch correction covariate


class BiologicalEncoder:
    """scVI-based encoder for compressing single-cell data into a latent state.

    The latent representation captures gene expression variation including
    aging-relevant signals, while correcting for batch effects.
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        self.model = None
        self._adata_setup = False

    def setup(self, adata: ad.AnnData):
        """Register AnnData with scVI and prepare for training."""
        import scvi

        scvi.model.SCVI.setup_anndata(
            adata,
            batch_key=self.config.batch_key,
        )
        self.model = scvi.model.SCVI(
            adata,
            n_latent=self.config.latent_dim,
            n_layers=self.config.n_layers,
            n_hidden=self.config.n_hidden,
            dropout_rate=self.config.dropout_rate,
            gene_likelihood=self.config.gene_likelihood,
        )
        self._adata_setup = True
        logger.info(
            f"scVI model initialized: latent_dim={self.config.latent_dim}, "
            f"n_layers={self.config.n_layers}, n_hidden={self.config.n_hidden}"
        )

    def train(self, adata: Optional[ad.AnnData] = None):
        """Train the scVI model."""
        if self.model is None:
            if adata is None:
                raise ValueError("Must call setup() first or provide adata")
            self.setup(adata)

        logger.info(f"Training scVI for max {self.config.max_epochs} epochs...")
        self.model.train(
            max_epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            early_stopping=self.config.early_stopping,
            early_stopping_patience=self.config.early_stopping_patience,
            plan_kwargs={"lr": self.config.learning_rate},
        )
        logger.info("scVI training complete")

    def encode(self, adata: ad.AnnData) -> np.ndarray:
        """Encode cells into latent space.

        Args:
            adata: AnnData with same var_names as training data.

        Returns:
            Latent embeddings array of shape (n_cells, latent_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.get_latent_representation(adata)

    def save(self, path: str | Path):
        """Save trained model to directory."""
        if self.model is None:
            raise RuntimeError("No model to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path), overwrite=True)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path, adata: ad.AnnData):
        """Load a trained model from directory."""
        import scvi

        path = Path(path)
        self.model = scvi.model.SCVI.load(str(path), adata=adata)
        logger.info(f"Model loaded from {path}")
