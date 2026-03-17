"""Hallmark of aging scoring heads (T-013, T-014).

Multi-task heads that score biological hallmarks from latent state.
Each hallmark gets an independent probe trained on gene-set-derived
pseudo-labels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class HallmarkConfig:
    """Configuration for hallmark scoring heads."""

    hidden_dim: int = 64
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_epochs: int = 100
    batch_size: int = 256
    patience: int = 10


def load_hallmark_gene_sets(path: str | Path) -> dict[str, list[str]]:
    """Load hallmark gene sets from JSON.

    Returns:
        Dict mapping hallmark name to list of gene symbols.
    """
    raw = json.loads(Path(path).read_text())
    return {name: info["genes"] for name, info in raw["hallmarks"].items()}


def compute_gene_set_scores(
    expression_matrix: np.ndarray,
    gene_names: list[str],
    gene_set: list[str],
) -> np.ndarray:
    """Compute mean z-score gene set enrichment (UCell-like).

    Args:
        expression_matrix: Log-normalized expression (n_cells, n_genes).
        gene_names: Gene names matching columns.
        gene_set: List of genes in the hallmark set.

    Returns:
        Score per cell (n_cells,).
    """
    gene_idx = [i for i, g in enumerate(gene_names) if g in set(gene_set)]
    if len(gene_idx) == 0:
        logger.warning(f"No genes matched from set of {len(gene_set)}")
        return np.zeros(expression_matrix.shape[0])

    subset = expression_matrix[:, gene_idx]

    # Z-score per gene, then mean across genes
    mean = subset.mean(axis=0, keepdims=True)
    std = subset.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z_scored = (subset - mean) / std
    scores = z_scored.mean(axis=1)

    return scores


class HallmarkHead(nn.Module):
    """Single hallmark scoring head — MLP probe on latent state."""

    def __init__(self, input_dim: int, config: HallmarkConfig | None = None):
        super().__init__()
        config = config or HallmarkConfig()
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HallmarkScorer:
    """Collection of hallmark heads for multi-hallmark scoring.

    Provides the `score_hallmarks(latent_state) -> dict[str, float]` API
    required by the reward function and observation space.
    """

    def __init__(
        self,
        hallmark_names: list[str],
        input_dim: int,
        config: HallmarkConfig | None = None,
    ):
        self.hallmark_names = hallmark_names
        self.input_dim = input_dim
        self.config = config or HallmarkConfig()
        self.heads: dict[str, HallmarkHead] = {
            name: HallmarkHead(input_dim, self.config) for name in hallmark_names
        }

    def score_hallmarks(self, latent_state: np.ndarray) -> dict[str, float]:
        """Score all hallmarks for a single latent state vector.

        Args:
            latent_state: Latent vector (latent_dim,) or (1, latent_dim).

        Returns:
            Dict mapping hallmark name to score.
        """
        if latent_state.ndim == 1:
            latent_state = latent_state[np.newaxis, :]

        z = torch.from_numpy(latent_state).float()
        scores = {}
        for name, head in self.heads.items():
            head.eval()
            with torch.no_grad():
                scores[name] = float(head(z).squeeze().item())
        return scores

    def score_hallmarks_batch(self, latent_states: np.ndarray) -> dict[str, np.ndarray]:
        """Score all hallmarks for a batch of latent states.

        Args:
            latent_states: Latent matrix (n_cells, latent_dim).

        Returns:
            Dict mapping hallmark name to score array (n_cells,).
        """
        z = torch.from_numpy(latent_states).float()
        scores = {}
        for name, head in self.heads.items():
            head.eval()
            with torch.no_grad():
                scores[name] = head(z).squeeze(-1).numpy()
        return scores

    def train_head(
        self,
        hallmark_name: str,
        latent_train: np.ndarray,
        scores_train: np.ndarray,
        latent_val: np.ndarray,
        scores_val: np.ndarray,
    ) -> dict:
        """Train a single hallmark head.

        Returns:
            Metrics dict with correlation and loss info.
        """
        head = self.heads[hallmark_name]
        optimizer = torch.optim.Adam(head.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.from_numpy(latent_train).float(),
            torch.from_numpy(scores_train).float(),
        )
        train_dl = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        z_val_t = torch.from_numpy(latent_val).float()
        y_val_t = torch.from_numpy(scores_val).float()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            head.train()
            for z_batch, y_batch in train_dl:
                optimizer.zero_grad()
                pred = head(z_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

            head.eval()
            with torch.no_grad():
                val_pred = head(z_val_t).squeeze(-1)
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        # Compute Spearman correlation
        from scipy.stats import spearmanr

        head.eval()
        with torch.no_grad():
            y_pred = head(z_val_t).squeeze(-1).numpy()
        rho, pval = spearmanr(scores_val, y_pred)

        metrics = {
            "spearman_rho": float(rho),
            "pval": float(pval),
            "val_loss": float(best_val_loss),
        }
        logger.info(f"Hallmark '{hallmark_name}': Spearman ρ={rho:.3f}, p={pval:.2e}")
        return metrics

    def save(self, path: str | Path):
        """Save all heads to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, head in self.heads.items():
            torch.save(head.state_dict(), path / f"{name}.pt")

    def load(self, path: str | Path):
        """Load all heads from a directory."""
        path = Path(path)
        for name, head in self.heads.items():
            state_path = path / f"{name}.pt"
            if state_path.exists():
                head.load_state_dict(torch.load(state_path, weights_only=True))
                logger.info(f"Loaded hallmark head: {name}")
