"""Tests for aging clock regression head (T-012)."""

import numpy as np

from gnoma.models.aging_clock import AgingClockConfig, AgingClockHead, train_aging_clock


def test_aging_clock_forward():
    model = AgingClockHead(input_dim=128)
    z = np.random.randn(10, 128).astype(np.float32)
    ages = model.predict_age(z)
    assert ages.shape == (10,)


def test_train_aging_clock_convergence():
    """Train on synthetic data with a clear linear age signal."""
    rng = np.random.RandomState(42)
    n_train, n_val = 200, 50
    latent_dim = 32

    # Create synthetic data where age correlates with first latent dim
    z_train = rng.randn(n_train, latent_dim).astype(np.float32)
    ages_train = z_train[:, 0] * 10 + 50 + rng.randn(n_train).astype(np.float32) * 2
    z_val = rng.randn(n_val, latent_dim).astype(np.float32)
    ages_val = z_val[:, 0] * 10 + 50 + rng.randn(n_val).astype(np.float32) * 2

    config = AgingClockConfig(
        hidden_dim=0,  # linear probe
        max_epochs=500,
        learning_rate=1e-2,
        patience=50,
        batch_size=64,
    )

    model, metrics = train_aging_clock(z_train, ages_train, z_val, ages_val, config)

    assert metrics["r2"] > 0.3  # should learn the linear relationship
    assert metrics["mae"] < 15.0


def test_mlp_aging_clock():
    model = AgingClockHead(input_dim=64, config=AgingClockConfig(hidden_dim=32))
    z = np.random.randn(5, 64).astype(np.float32)
    ages = model.predict_age(z)
    assert ages.shape == (5,)
