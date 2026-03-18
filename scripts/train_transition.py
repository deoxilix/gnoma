#!/usr/bin/env python3
"""Train neural transition model on Perturb-seq data (Sprint 6).

Encodes control and perturbed cells through the trained scVI encoder,
computes pseudo-bulk centroids, and trains the TransitionNetwork to
predict state deltas from intervention embeddings.

Expects:
  - data/perturbseq/ReplogleWeissman2022_K562_essential.h5ad
  - data/processed/train.h5ad (for HVG list and gene mapping)
  - models/encoder/ (trained scVI model)
"""

import json
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def align_genes(adata_ps, adata_ref):
    """Align Perturb-seq genes to reference HVG set.

    Returns a new AnnData with the same genes as the reference (HVG subset),
    zero-filling any missing genes.
    """
    # Get reference gene names (HVGs only)
    if "feature_name" in adata_ref.var.columns:
        ref_gene_names = list(adata_ref.var["feature_name"])
    else:
        ref_gene_names = list(adata_ref.var_names)

    hvg_mask = adata_ref.var.get("highly_variable")
    if hvg_mask is not None:
        if "feature_name" in adata_ref.var.columns:
            hvg_names = list(adata_ref.var["feature_name"][hvg_mask])
        else:
            hvg_names = list(adata_ref.var_names[hvg_mask])
    else:
        hvg_names = ref_gene_names

    logger.info(f"Reference HVGs: {len(hvg_names)}")

    # Get Perturb-seq gene names
    ps_gene_names = list(adata_ps.var_names)

    # Find overlap
    hvg_set = set(hvg_names)
    ps_set = set(ps_gene_names)
    overlap = hvg_set & ps_set
    logger.info(f"Gene overlap: {len(overlap)} / {len(hvg_names)}")

    # Build aligned matrix
    X_ps = adata_ps.X
    if scipy.sparse.issparse(X_ps):
        X_ps = X_ps.toarray()

    # Create zero-filled aligned matrix
    n_cells = adata_ps.n_obs
    n_hvg = len(hvg_names)
    X_aligned = np.zeros((n_cells, n_hvg), dtype=np.float32)

    ps_name_to_idx = {name: idx for idx, name in enumerate(ps_gene_names)}

    matched = 0
    for i, gene in enumerate(hvg_names):
        if gene in ps_name_to_idx:
            X_aligned[:, i] = X_ps[:, ps_name_to_idx[gene]]
            matched += 1

    logger.info(f"Aligned {matched}/{n_hvg} genes (rest zero-filled)")

    # Create aligned AnnData
    import pandas as pd
    var_df = pd.DataFrame(index=hvg_names)
    adata_aligned = ad.AnnData(
        X=X_aligned,
        obs=adata_ps.obs.copy(),
        var=var_df,
    )

    return adata_aligned


def encode_with_scvi(adata_aligned, model_dir="models/encoder"):
    """Encode aligned cells through trained scVI encoder."""
    import scvi

    # Load the trained scVI model
    # We need an adata with matching var to the training data
    logger.info(f"Loading scVI model from {model_dir}")

    # Load the reference training data to get the correct setup
    adata_ref = ad.read_h5ad("data/processed/train.h5ad")

    # Get HVG subset
    hvg_mask = adata_ref.var.get("highly_variable")
    if hvg_mask is not None:
        adata_ref_hvg = adata_ref[:, hvg_mask].copy()
    else:
        adata_ref_hvg = adata_ref.copy()

    # Setup the reference first to register the model
    scvi.model.SCVI.setup_anndata(adata_ref_hvg)

    # Load saved model
    model = scvi.model.SCVI.load(model_dir, adata=adata_ref_hvg)

    # Now encode the Perturb-seq data
    # The aligned adata has the same var as adata_ref_hvg
    # We need to make it compatible with the scVI model
    logger.info(f"Encoding {adata_aligned.n_obs} Perturb-seq cells...")

    # scVI expects the adata to be registered. We can use the model's transform approach.
    # Register the Perturb-seq data
    scvi.model.SCVI.setup_anndata(adata_aligned)

    # Get latent representation in batches
    batch_size = 1024
    latents = []
    n_cells = adata_aligned.n_obs

    model.module.eval()
    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch_data = torch.from_numpy(adata_aligned.X[start:end]).float()

            # Use the encoder directly
            # scVI model structure: module.z_encoder
            inference_inputs = {"x": batch_data}
            # Add batch and other required fields
            batch_idx = torch.zeros(end - start, 1, dtype=torch.long)
            cat_covs = None

            # Direct encoding through the module
            qz = model.module.z_encoder(batch_data, batch_idx)
            z = qz["qz"].loc  # Mean of the latent distribution
            latents.append(z.cpu().numpy())

    latent = np.concatenate(latents, axis=0)
    logger.info(f"Encoded latent shape: {latent.shape}")
    return latent


def encode_simple(adata_aligned, model_dir="models/encoder"):
    """Encode aligned cells through trained scVI encoder using direct z_encoder access."""
    import scvi

    logger.info(f"Loading scVI model from {model_dir}")

    # Load reference data for model loading
    adata_ref = ad.read_h5ad("data/processed/train.h5ad")
    hvg_mask = adata_ref.var.get("highly_variable")
    if hvg_mask is not None:
        adata_ref_hvg = adata_ref[:, hvg_mask].copy()
    else:
        adata_ref_hvg = adata_ref.copy()

    scvi.model.SCVI.setup_anndata(adata_ref_hvg)
    model = scvi.model.SCVI.load(model_dir, adata=adata_ref_hvg)

    logger.info(f"Encoding {adata_aligned.n_obs} cells through scVI encoder...")
    model.module.eval()

    X = adata_aligned.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X_tensor = torch.from_numpy(X).float()

    batch_size = 512
    latents = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            end = min(start + batch_size, len(X_tensor))
            batch = X_tensor[start:end]
            batch_idx = torch.zeros(end - start, 1, dtype=torch.long)

            # scVI z_encoder returns (Normal distribution, latent_library)
            encoder_input = torch.log1p(batch)
            qz_out = model.module.z_encoder(encoder_input, batch_idx)
            # qz_out is a tuple: (Normal dist, library tensor)
            z = qz_out[0].loc  # Mean of the Normal distribution
            latents.append(z.cpu().numpy())

    latent = np.concatenate(latents, axis=0)
    logger.info(f"Encoded latent shape: {latent.shape}")
    return latent


def compute_pseudobulk_centroids(latent, perturbation_labels, control_label="control"):
    """Compute mean latent centroid per perturbation and for controls."""
    labels = np.array(perturbation_labels)
    unique_perts = sorted(set(labels) - {control_label})

    # Control centroid
    control_mask = labels == control_label
    control_centroid = latent[control_mask].mean(axis=0)
    logger.info(f"Control centroid: {control_mask.sum()} cells")

    # Per-perturbation centroids and deltas
    centroids = {}
    deltas = {}
    for pert in unique_perts:
        mask = labels == pert
        n = mask.sum()
        if n < 5:  # skip very low-count perturbations
            continue
        centroid = latent[mask].mean(axis=0)
        delta = centroid - control_centroid
        centroids[pert] = centroid
        deltas[pert] = delta

    logger.info(f"Computed centroids for {len(centroids)} perturbations (min 5 cells)")
    return control_centroid, centroids, deltas


def main():
    ps_path = Path("data/perturbseq/ReplogleWeissman2022_K562_essential.h5ad")
    if not ps_path.exists():
        logger.error(f"Missing: {ps_path}. Run download_perturbseq.py first.")
        sys.exit(1)

    # --- Load Perturb-seq data ---
    logger.info("Loading Perturb-seq data...")
    adata_ps = ad.read_h5ad(ps_path)
    logger.info(f"Perturb-seq shape: {adata_ps.shape}")

    # --- Load reference training data for gene alignment ---
    adata_ref = ad.read_h5ad("data/processed/train.h5ad")

    # --- Align genes ---
    logger.info("Aligning Perturb-seq genes to reference HVGs...")
    adata_aligned = align_genes(adata_ps, adata_ref)
    del adata_ref  # free memory

    # --- Encode through scVI ---
    try:
        latent = encode_simple(adata_aligned, model_dir="models/encoder")
    except Exception as e:
        logger.warning(f"scVI encoding failed: {e}")
        logger.info("Falling back to PCA-based encoding...")
        # Fallback: use PCA projection matched to the latent dim
        from sklearn.decomposition import PCA

        X = adata_aligned.X
        if scipy.sparse.issparse(X):
            X = X.toarray()
        pca = PCA(n_components=128, random_state=42)
        latent = pca.fit_transform(X).astype(np.float32)
        logger.info(f"PCA latent shape: {latent.shape}")

    # --- Compute pseudo-bulk centroids ---
    perturbation_labels = adata_ps.obs["perturbation"].values
    control_centroid, centroids, deltas = compute_pseudobulk_centroids(
        latent, perturbation_labels, control_label="control"
    )

    # --- Prepare training data ---
    # For each perturbation: (control_state, intervention_embedding) → delta
    # We'll learn intervention embeddings jointly with the transition model

    pert_names = sorted(deltas.keys())
    n_perts = len(pert_names)
    latent_dim = latent.shape[1]
    logger.info(f"Training with {n_perts} perturbations, latent dim {latent_dim}")

    # Create training pairs: use per-cell data, not just centroids, for more training signal
    # For each perturbed cell: (control_centroid, perturbation_id) → (cell_latent - control_centroid)
    train_states = []
    train_pert_ids = []
    train_deltas = []

    pert_to_idx = {p: i for i, p in enumerate(pert_names)}

    for pert in pert_names:
        mask = perturbation_labels == pert
        cell_latents = latent[mask]
        cell_deltas = cell_latents - control_centroid

        for i in range(len(cell_latents)):
            train_states.append(control_centroid)
            train_pert_ids.append(pert_to_idx[pert])
            train_deltas.append(cell_deltas[i])

    train_states = np.array(train_states, dtype=np.float32)
    train_pert_ids = np.array(train_pert_ids, dtype=np.int64)
    train_deltas = np.array(train_deltas, dtype=np.float32)

    logger.info(f"Training samples: {len(train_states)}")

    # --- Split train/val ---
    rng = np.random.RandomState(42)
    n = len(train_states)
    idx = rng.permutation(n)
    val_frac = 0.15
    n_val = int(n * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    # --- Build model ---
    from gnoma.simulator.transition_model import TransitionConfig, TransitionNetwork

    embed_dim = 64
    config = TransitionConfig(
        latent_dim=latent_dim,
        intervention_embed_dim=embed_dim,
        hidden_dim=256,
        n_layers=3,
        dropout=0.1,
        learning_rate=1e-3,
        max_epochs=200,
        batch_size=256,
        patience=20,
    )

    transition_net = TransitionNetwork(config)

    # Learnable intervention embeddings
    intervention_embeddings = nn.Embedding(n_perts, embed_dim)
    nn.init.normal_(intervention_embeddings.weight, std=0.1)

    # --- Training loop ---
    optimizer = torch.optim.Adam(
        list(transition_net.parameters()) + list(intervention_embeddings.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Convert to tensors
    states_t = torch.from_numpy(train_states)
    pert_ids_t = torch.from_numpy(train_pert_ids)
    deltas_t = torch.from_numpy(train_deltas)

    train_dataset = TensorDataset(states_t[train_idx], pert_ids_t[train_idx], deltas_t[train_idx])
    val_dataset = TensorDataset(states_t[val_idx], pert_ids_t[val_idx], deltas_t[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    logger.info(f"Training transition model ({config.max_epochs} epochs, {len(train_idx)} train, {len(val_idx)} val)...")

    for epoch in range(config.max_epochs):
        # Train
        transition_net.train()
        train_loss = 0.0
        n_batches = 0

        for batch_states, batch_perts, batch_deltas in train_loader:
            optimizer.zero_grad()
            emb = intervention_embeddings(batch_perts)
            pred_delta, pred_log_var = transition_net(batch_states, emb)

            # Heteroscedastic Gaussian NLL loss
            var = torch.exp(pred_log_var).clamp(min=1e-6)
            nll = 0.5 * (pred_log_var + (batch_deltas - pred_delta) ** 2 / var)
            loss = nll.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(transition_net.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validate
        transition_net.eval()
        val_loss = 0.0
        val_mse = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch_states, batch_perts, batch_deltas in val_loader:
                emb = intervention_embeddings(batch_perts)
                pred_delta, pred_log_var = transition_net(batch_states, emb)
                var = torch.exp(pred_log_var).clamp(min=1e-6)
                nll = 0.5 * (pred_log_var + (batch_deltas - pred_delta) ** 2 / var)
                val_loss += nll.mean().item()
                val_mse += ((batch_deltas - pred_delta) ** 2).mean().item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        val_mse /= max(n_val_batches, 1)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == config.max_epochs - 1:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:3d}: train_nll={train_loss:.4f}, val_nll={val_loss:.4f}, "
                f"val_mse={val_mse:.4f}, lr={lr:.1e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                "transition_net": transition_net.state_dict(),
                "intervention_embeddings": intervention_embeddings.state_dict(),
            }
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state:
        transition_net.load_state_dict(best_state["transition_net"])
        intervention_embeddings.load_state_dict(best_state["intervention_embeddings"])

    # --- Evaluate on validation set ---
    transition_net.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_states, batch_perts, batch_deltas in val_loader:
            emb = intervention_embeddings(batch_perts)
            pred_delta, _ = transition_net(batch_states, emb)
            all_preds.append(pred_delta.numpy())
            all_targets.append(batch_deltas.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Per-dimension correlation
    from scipy.stats import pearsonr

    dim_corrs = []
    for d in range(latent_dim):
        if all_targets[:, d].std() > 1e-8:
            r, _ = pearsonr(all_preds[:, d], all_targets[:, d])
            dim_corrs.append(r)
    mean_corr = np.mean(dim_corrs)
    median_corr = np.median(dim_corrs)

    # Per-perturbation correlation (average across dimensions)
    pert_corrs = {}
    val_pert_ids = train_pert_ids[val_idx]
    for p_idx, p_name in enumerate(pert_names):
        mask = val_pert_ids == p_idx
        if mask.sum() < 5:
            continue
        pred_centroid = all_preds[mask].mean(axis=0)
        true_centroid = all_targets[mask].mean(axis=0)
        r, _ = pearsonr(pred_centroid, true_centroid)
        pert_corrs[p_name] = float(r)

    if pert_corrs:
        mean_pert_corr = np.mean(list(pert_corrs.values()))
    else:
        mean_pert_corr = 0.0

    logger.info(f"Val per-dim correlation: mean={mean_corr:.3f}, median={median_corr:.3f}")
    logger.info(f"Val per-perturbation centroid corr: mean={mean_pert_corr:.3f}")

    # --- Save model ---
    model_dir = Path("models/transition")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save perturbation name mapping
    pert_mapping = {name: idx for idx, name in enumerate(pert_names)}

    # Map matched ontology interventions to their learned embeddings
    ontology = json.loads(Path("data/interventions/ontology_v1.json").read_text())
    ontology_gene_map = {}
    for intervention in ontology["interventions"]:
        gene = intervention.get("target_gene")
        if gene and gene in pert_to_idx:
            ontology_gene_map[intervention["id"]] = {
                "gene": gene,
                "pert_idx": pert_to_idx[gene],
            }

    save_dict = {
        "config": config.__dict__,
        "transition_net_state": best_state["transition_net"] if best_state else transition_net.state_dict(),
        "intervention_embeddings_state": best_state["intervention_embeddings"] if best_state else intervention_embeddings.state_dict(),
        "pert_names": pert_names,
        "pert_to_idx": pert_to_idx,
        "ontology_gene_map": ontology_gene_map,
        "control_centroid": control_centroid,
        "metrics": {
            "best_val_nll": float(best_val_loss),
            "val_mse": float(((all_preds - all_targets) ** 2).mean()),
            "mean_dim_correlation": float(mean_corr),
            "median_dim_correlation": float(median_corr),
            "mean_perturbation_centroid_correlation": float(mean_pert_corr),
            "n_perturbations": n_perts,
            "n_train_samples": len(train_idx),
            "n_val_samples": len(val_idx),
            "n_ontology_matched": len(ontology_gene_map),
        },
    }
    torch.save(save_dict, model_dir / "transition_model.pt")
    logger.info(f"Saved transition model to {model_dir}")

    # Save metrics as JSON too
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(save_dict["metrics"], f, indent=2)

    print(f"\n=== Transition Model Training Summary ===")
    print(f"Perturbations: {n_perts}")
    print(f"Training samples: {len(train_idx):,}")
    print(f"Best val NLL: {best_val_loss:.4f}")
    print(f"Val MSE: {((all_preds - all_targets) ** 2).mean():.4f}")
    print(f"Per-dim correlation: mean={mean_corr:.3f}, median={median_corr:.3f}")
    print(f"Per-perturbation centroid corr: mean={mean_pert_corr:.3f}")
    print(f"Ontology-matched perturbations: {len(ontology_gene_map)}")
    if ontology_gene_map:
        print(f"  Genes: {', '.join(m['gene'] for m in ontology_gene_map.values())}")
    print(f"\nTop-10 best predicted perturbations:")
    for name, r in sorted(pert_corrs.items(), key=lambda x: x[1], reverse=True)[:10]:
        flag = " [ONTOLOGY]" if name in pert_to_idx and any(
            m["gene"] == name for m in ontology_gene_map.values()
        ) else ""
        print(f"  {name}: r={r:.3f}{flag}")


if __name__ == "__main__":
    main()
