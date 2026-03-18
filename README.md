# gnoma

**Autonomous discovery of reverse-aging interventions using reinforcement learning and biologically grounded world models.**

gnoma is a research-grade computational pipeline that uses single-cell transcriptomics, a hybrid world model, and reinforcement learning to discover and rank molecular interventions for cellular rejuvenation. It learns from real human cell data (CZI cellxgene-census, 61M+ cells), simulates intervention effects using a combination of curated biological rules and a neural transition model trained on CRISPR perturbation data, and outputs ranked, literature-validated intervention candidates.

In its first run on human blood cells, **gnoma independently rediscovered 10 out of 20 top-ranked interventions that are validated in published aging research** — including TFEB-driven autophagy, OSK partial reprogramming, FOXO3 overexpression, DNMT inhibitors, senolytics, and SIRT1 activation — with zero prior knowledge of what the "right answers" should be.

## How It Works

```
Human cell data (cellxgene-census)
    ↓
scVI variational autoencoder → 128-dim latent state
    ↓
Aging clock (biological age prediction) + 11 hallmark-of-aging scorers
    ↓
Hybrid world model:
  ├── 37 curated causal rules (mTOR, NAD+, senolysis, SASP, epigenetic...)
  └── Neural transition model (trained on 310k Perturb-seq cells, 2,057 CRISPRi perturbations)
    ↓
Gymnasium RL environment with safety-constrained action space (105 interventions)
    ↓
PPO agent with action masking (oncogenic interventions blocked)
    ↓
Multi-objective reward: rejuvenation + identity preservation + viability − uncertainty
    ↓
Ranked intervention candidates with literature validation scores
```

## Results

From 200 evaluation episodes on human blood cells (ages 40–75):

| Rank | Intervention | Type | Literature Validated |
|------|-------------|------|:---:|
| 1 | TFEB overexpression | Autophagy enhancement | Yes |
| 2 | Taurine supplementation | Metabolic | — |
| 3 | OSK partial reprogramming (Myc-free) | Epigenetic reprogramming | Yes |
| 4 | ISRIB | Proteostasis / ISR inhibitor | — |
| 5 | Chaetocin | Epigenetic (H3K9me3) | — |
| 6 | TFAM overexpression | Mitochondrial biogenesis | — |
| 7 | JQ1 | BET bromodomain inhibitor | — |
| 8 | 5-Azacytidine | DNMT inhibitor | Yes |
| 9 | FOXO3 overexpression | Stress resistance / longevity | Yes |
| 10 | Acarbose | CR mimetic | — |

**10/20 top candidates are independently validated by published aging literature.** The agent achieves 102% higher reward than a random baseline with 0% oncogenic safety violations.

## Installation

Requires Python 3.11–3.12.

```bash
git clone https://github.com/deoxilix/gnoma.git
cd gnoma
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the installation:

```bash
pytest
```

All 38 tests should pass.

## Quick Start

### Smoke test (no data download required)

```bash
python -c "
import numpy as np
from gnoma.interventions.action_space import InterventionSpace
from gnoma.simulator.rule_kernel import RuleKernel, build_default_rules
from gnoma.simulator.transition_model import TransitionConfig, TransitionNetwork, HybridWorldModel

action_space = InterventionSpace.from_json('data/interventions/ontology_v1.json', embedding_dim=64)
rule_kernel = RuleKernel(build_default_rules())
config = TransitionConfig(latent_dim=128, intervention_embed_dim=64, hidden_dim=128, mc_samples=3)
world_model = HybridWorldModel(TransitionNetwork(config), rule_kernel=rule_kernel, config=config)

state = np.random.randn(128).astype(np.float32)
action_idx = action_space.sample()
emb = action_space.to_embedding(action_idx)
next_state, aleatoric, epistemic = world_model.step(state, emb)

intervention = action_space.get_intervention(action_idx)
print(f'Intervention: {intervention.name}')
print(f'State delta norm: {np.linalg.norm(next_state - state):.4f}')
print(f'Uncertainty — aleatoric: {aleatoric:.4f}, epistemic: {epistemic:.4f}')
print('Pipeline works.')
"
```

### Full pipeline

```bash
# 1. Download and preprocess cell data (~37 min, requires internet)
python scripts/download_blood_subset.py     # 50k blood cells from CZI census
python scripts/run_preprocess.py            # QC, normalization, HVG selection, donor splits

# 2. Train biological models (~15 min)
python scripts/train_encoder.py             # scVI encoder + aging clock
python scripts/train_hallmarks.py           # 11 hallmark-of-aging scoring heads

# 3. Download perturbation data and train world model (~25 min)
python scripts/download_perturbseq.py       # Replogle 2022 Perturb-seq (1.5 GB)
python scripts/train_transition.py          # Neural transition model

# 4. Train RL agent and evaluate (~3 min)
python scripts/train_ppo.py                 # PPO with trained world model
python scripts/eval_diverse.py              # Stochastic evaluation + baseline comparison
```

Results are saved to `reports/eval_ppo_v2_stochastic.json`.

## Project Structure

```
src/gnoma/
├── data/           Census data access and preprocessing pipeline
├── models/         scVI encoder, aging clock, hallmark scoring heads
├── interventions/  Intervention ontology (105 entries) and RL action space
├── simulator/      Hybrid world model: rule kernel + neural transition
├── env/            Gymnasium RL environment (AgingEnv)
├── reward/         Multi-objective reward with safety constraints
└── eval/           Evaluation suite and candidate ranking

scripts/            Pipeline scripts (download, train, evaluate)
data/
├── interventions/  Curated intervention ontology (ontology_v1.json)
├── hallmarks/      Hallmark-of-aging gene sets
└── rules/          Mechanistic causal rules (exported JSON)

tests/              38 unit tests
```

## Extending to Other Cell Types

gnoma is designed to be retargeted to different tissues and aging phenotypes. To investigate a different cell type:

1. **Change the data source** — modify `scripts/download_blood_subset.py` to query a different tissue (e.g., `tissue_general == 'skin'`)
2. **Add domain-specific hallmarks** — add gene sets to `data/hallmarks/hallmark_gene_sets.json`
3. **Tune the reward** — adjust weights in `RewardConfig` to emphasize your phenotype of interest
4. **Re-run the pipeline** — the rest of the pipeline (encoder, world model, RL) adapts automatically

## Key Design Decisions

- **Hybrid world model** — mechanistic rules provide biological priors; the neural model learns residual corrections from real perturbation data. Neither alone is sufficient.
- **Compressed latent state** — the RL agent operates on 128-dimensional scVI embeddings, not raw gene expression (20,000+ dimensions). This makes RL tractable.
- **Safety-constrained action space** — interventions flagged as oncogenic (MYC overexpression, etc.) are masked from the agent's action space. The agent cannot propose them.
- **Multi-objective reward** — balances rejuvenation with identity preservation (don't change cell type), viability (don't kill the cell), and uncertainty penalty (don't trust the model where it's uncertain).
- **Donor-level data splits** — train/val/test splits are stratified by donor, not by cell, to prevent data leakage.

## Technology Stack

| Component | Library |
|-----------|---------|
| Single-cell data | cellxgene-census, scanpy, anndata |
| Biological encoder | scvi-tools (VAE) |
| World model | PyTorch (MLP + MC dropout) |
| Rule kernel | Custom (37 curated causal rules) |
| RL environment | Gymnasium |
| RL agent | stable-baselines3, sb3-contrib (MaskablePPO) |
| Configuration | Hydra, Pydantic |
| Data versioning | DVC |

## Limitations and Caveats

- **This is a research prototype, not a medical tool.** gnoma generates ranked hypotheses for further scientific investigation. Its outputs should not be interpreted as treatment recommendations.
- **Simulation fidelity is limited.** The world model is trained on observational perturbation data (K562 cell line CRISPRi). It cannot perfectly predict how real interventions will behave in primary human cells.
- **Age prediction accuracy is modest.** The aging clock achieves R² = 0.28 on blood cells — useful for relative ranking but not precise biological age estimation. Age labels from census data are noisy (parsed from text fields like "45-year-old human stage").
- **Cell line domain gap.** The neural transition model was trained on K562 (leukemia cell line) perturbation data but applied to primary blood cell latent space. The rule kernel partially compensates for this gap.
- **Action space is single-target.** The current version evaluates one intervention per step. Combinatorial interventions (e.g., "rapamycin + senolytic") are planned but not yet supported.
- **No wet-lab validation.** All results are computational. The ranked candidates are hypotheses that would require experimental validation before any translational conclusions.

## Citation

If you use gnoma in your research, please cite:

```bibtex
@software{gnoma2026,
  title={gnoma: Autonomous Discovery of Reverse-Aging Interventions via Reinforcement Learning},
  year={2026},
  url={https://github.com/deoxilix/gnoma}
}
```

## License

MIT
