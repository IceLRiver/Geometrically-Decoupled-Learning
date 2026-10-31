# Geometrically Decoupled Learning

Data and model code for **Geometrically Decoupled Learning for Space Target Motion State Estimation from Short-Arc Angles-Only Observations**.

## Datasets

| Collection | Arcs | Train / validation / test |
|---|---:|---|
| Sentinel-1 precise-orbit benchmark | 9,632 | 4,815 / 2,412 / 2,405 |
| Multiregime simulation benchmark | 120,000 | 79,980 / 19,995 / 20,025 |

Each input `X` has shape `(N,40,6)`: observer position in km followed by the unit line-of-sight vector, sampled every second. Labels `y` have shape `(N,6)`: target position (km) and velocity (km/s) at the last epoch. Dataset partitions use disjoint target identities. The simulation collection covers LEO, MEO and high-altitude orbits equally, with four angular-noise levels. The Sentinel-1 benchmark uses public precise ephemerides and reconstructed angular observations. Contains modified Copernicus Sentinel data (2021, 2026); [source catalogue](https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/).

The paper evaluates all primary methods on the frozen 300 public, 300 simulation and 122 real-observation arcs. The simulation test cohort includes 100 arcs per regime. Aggregate results for all three sources are in `evidence/v4/results.csv`; public/simulation predictions and exact test indices accompany them. Sensor records are subject to the data owner's access restrictions.

## Installation

Neural training/inference requires Python 3.9 or later:

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

Classical/physics and RBF baselines use a separate Python 3.11 environment with `requirements-orbits.txt`.

## Evaluation

```bash
python scripts/v4/evaluate_checkpoint.py --dataset public --checkpoint checkpoints/v4/public__GDL.pt
python scripts/v4/evaluate_checkpoint.py --dataset simulation --checkpoint checkpoints/v4/simulation__GDL.pt
python scripts/v4/evaluate_physics.py --dataset public --method alpha-IOD-2025 --workers 4
```

`models/geometric_v4.py` exposes `load_checkpoint` and observation-only `predict`. Primary configurations were selected on validation data and frozen before final test scoring. GDL uses the KAN backbone for public/real observations and the MLP backbone for the broad simulation benchmark. `evidence/v4/selected_models.json` records configurations, validation scores and checkpoint hashes. Published physics baselines are research reimplementations with observation-cadence and near-Earth adaptations detailed in the manuscript; they do not represent execution of the original authors' software.

## Training

```bash
python scripts/v4/train_models.py --dataset public --kind Geometric-KAN --coordinates observer --epochs 150
python scripts/v4/train_models.py --dataset simulation --kind Geometric-MLP --coordinates observer --epochs 100 --batch-size 512
python scripts/v4/rbf_baseline.py --lvlh --minmax
python scripts/v4/train_matched_ablation.py
```

Outputs go to `runs/v4/`. The primary framework predicts range, reconstructs position on the observed LOS, and predicts velocity conditioned on that reconstruction. The KAN instance combines adaptive periodic bases with a gated residual network. Matched-capacity ablations use shared training budgets; their full results are included with the primary comparisons.

Original observation arrays and the original architecture remain in `data/legacy_simulation/` and `models/orbit_physkan.py`.
