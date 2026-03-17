"""Aging clock regression head (T-012).

Linear or shallow MLP probe on frozen latent embeddings
to predict donor chronological age. Serves as the primary
aging trajectory signal for reward computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class AgingClockConfig:
    """Configuration for aging clock head."""

    hidden_dim: int = 0  # 0 = linear probe, >0 = MLP
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_epochs: int = 100
    batch_size: int = 256
    patience: int = 10


class AgingClockHead(nn.Module):
    """Regression head predicting chronological age from latent state."""

    def __init__(self, input_dim: int, config: AgingClockConfig | None = None):
        super().__init__()
        self.config = config or AgingClockConfig()

        if self.config.hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, 1),
            )
        else:
            # Linear probe
            self.net = nn.Linear(input_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict age from latent vector.

        Args:
            z: Latent embeddings (batch, latent_dim).

        Returns:
            Predicted age (batch, 1).
        """
        return self.net(z)

    def predict_age(self, z: np.ndarray) -> np.ndarray:
        """Numpy convenience — predict ages from latent vectors.

        If the model was trained with age normalization (via train_aging_clock),
        predictions are automatically denormalized to real age units.
        """
        self.eval()
        with torch.no_grad():
            z_t = torch.from_numpy(z).float()
            ages = self.forward(z_t).squeeze(-1).numpy()
        # Denormalize if normalization params are stored
        if hasattr(self, "_age_mean") and hasattr(self, "_age_std"):
            ages = ages * self._age_std + self._age_mean
        return ages


def train_aging_clock(
    latent_train: np.ndarray,
    ages_train: np.ndarray,
    latent_val: np.ndarray,
    ages_val: np.ndarray,
    config: AgingClockConfig | None = None,
) -> tuple[AgingClockHead, dict]:
    """Train aging clock head on frozen latent embeddings.

    Args:
        latent_train: Training latent vectors (n_cells, latent_dim).
        ages_train: Training ages (n_cells,).
        latent_val: Validation latent vectors.
        ages_val: Validation ages.
        config: Training config.

    Returns:
        Tuple of (trained model, metrics dict with r2, mae, train_losses, val_losses).
    """
    config = config or AgingClockConfig()
    input_dim = latent_train.shape[1]
    model = AgingClockHead(input_dim, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Normalize ages for stable training
    age_mean = float(ages_train.mean())
    age_std = float(ages_train.std())
    if age_std == 0:
        age_std = 1.0
    ages_train_norm = (ages_train - age_mean) / age_std
    ages_val_norm = (ages_val - age_mean) / age_std

    # Store normalization params on model for inference
    model._age_mean = age_mean
    model._age_std = age_std

    # Build data loaders
    train_ds = TensorDataset(
        torch.from_numpy(latent_train).float(),
        torch.from_numpy(ages_train_norm).float(),
    )
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    z_val_t = torch.from_numpy(latent_val).float()
    y_val_t = torch.from_numpy(ages_val_norm).float()

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(config.max_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for z_batch, y_batch in train_dl:
            optimizer.zero_grad()
            pred = model(z_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(z_batch)
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(z_val_t).squeeze(-1)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute final metrics (denormalize back to age space)
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(z_val_t).squeeze(-1).numpy()
    y_pred = y_pred_norm * age_std + age_mean
    y_true = ages_val

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(y_true - y_pred))

    metrics = {
        "r2": float(r2),
        "mae": float(mae),
        "best_val_loss": float(best_val_loss),
        "epochs_trained": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    logger.info(f"Aging clock trained: R²={r2:.3f}, MAE={mae:.1f} years")
    return model, metrics
