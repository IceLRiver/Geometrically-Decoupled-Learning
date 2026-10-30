# Geometrically Decoupled Learning

Data and model code for **Geometrically Decoupled Learning for Space Target Motion State Estimation from Short-Arc Angles-Only Observations**.

## Contents

| Component | Location |
|---|---|
| Geometrically decoupled model | `models/orbit_physkan.py` |
| Model training and evaluation | `scripts/run_learning.py`, `scripts/evaluate_saved.py` |
| Original model checkpoint | `checkpoints/original_orbit_physkan.pth` |
| Observation dataset: 236,575 arcs | `data/legacy_simulation/` |
| Compact evaluation data: 2,250 arcs | `evidence/synthetic_arcs.npz` |
| LEO/MEO/high-altitude coverage data: 120,000 arcs | `data/multiregime/` |
| Trained models and predictions | `evidence/synthetic_checkpoints/` |

## Data format

Each arc contains 40 observations. Input features are observer position followed by the three components of the line-of-sight unit vector. Labels are target position and velocity at the final observation epoch. Positions are in km and velocities in km/s.

The compact evaluation data store `X` as `(N,40,6)` and `y` as `(N,6)`. Dataset blocks in `data/legacy_simulation/` store flattened inputs `(N,240)`. The compact evaluation data have 1,554 training, 228 validation and 468 test arcs; target orbits are disjoint across these partitions. Partition files and normalization statistics accompany the arrays.

The coverage dataset balances 40,000 arcs per orbital regime and provides target-disjoint training/validation/test partitions of 79,980/19,995/20,025 arcs. Its blocks additionally store `orbit_id`, `regime`, `split`, and `noise_arcsec`; codes and counts are listed in its manifest. The saved compact model results refer to the compact evaluation data.

## Installation

Use Python 3.9 or later and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Evaluation

```bash
python scripts/verify_release.py
python scripts/evaluate_saved.py --method Geometric-KAN --seed 42
```

## Training

```bash
python scripts/run_learning.py --dataset synthetic --epochs 120
python scripts/run_spline2025.py --dataset synthetic
```

## Model

The model predicts range, reconstructs position on the measured line of sight, and estimates velocity conditioned on the reconstructed position. The orbit-physics KAN uses learnable periodic bases and a gated residual network. `models/orbit_physkan.py` contains the width-128 architecture; `scripts/run_learning.py` includes the compact model and its ablations.
