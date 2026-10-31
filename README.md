# Geometrically Decoupled Learning

Data and model code for **Geometrically Decoupled Learning for Space Target Motion State Estimation from Short-Arc Angles-Only Observations**.

## Data

| Collection / protocol | Arcs | Train / validation / test |
|---|---:|---|
| Sentinel-1 precise-orbit benchmark | 9,632 | 4,815 / 2,412 / 2,405; satellite-disjoint |
| Original simulation, Sim-H | 236,575 | 170,334 / 18,926 / 47,315; original sample split |
| Multiregime simulation, Sim-X | 120,000 | 79,980 / 19,995 / 20,025; orbit-disjoint |

Each input `X` has shape `(N,40,6)`: observer position in km and unit LOS at one-second cadence. Labels `y` contain terminal position (km) and velocity (km/s). Original arrays in `data/legacy_simulation` are stored flattened as `(N,240)`. Sim-X balances 40,000 LEO, MEO and high-altitude arcs. The Sentinel-1 collection uses reconstructed angles and precise ephemeris references. Contains modified Copernicus Sentinel data (2021, 2026); [source catalogue](https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/).

`data/benchmark_protocol_v5.json` fixes the common 300 public, 300 Sim-H, 300 Sim-X and 122 real test indices. Historical checkpoint reuse is recorded for Sim-H; the cross-orbit test remains separate. `data/controlled_angle_noise` contains the paired 20-arc/four-level measurement-sensitivity collection. Real sensor records remain subject to the data provider's access conditions.

## Installation

Use Python 3.9–3.11 for neural models:

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

Numerical orbit/RBF solvers use a separate Python 3.11 environment with `requirements-orbits.txt`.

## Evaluation

```bash
python scripts/v5/evaluate_checkpoint.py --dataset public --method GDL
python scripts/v5/evaluate_checkpoint.py --dataset legacy --method GDL --full-test
python scripts/v5/evaluate_checkpoint.py --dataset simulation --method B-spline-KAN-2025
python scripts/v5/evaluate_checkpoint.py --dataset public --method RGA-KAN-2026
python scripts/v5/evaluate_physics.py --dataset simulation --method XGEO-PIOD --workers 4
```

`models/geometric_v5.py` exposes `load_model` and `predict_states`; inference takes observations only. Training uses FP32; released inference uses FP64 to stabilize calibrated basis sums. Current metrics, selected settings and saved public/simulation predictions are in `evidence/v5/`. All ten methods use identical comparison arcs. PIOD denotes the 2023 method; XGEO-PIOD denotes the 2026 extension. Physics-based rows are documented task reimplementations. The additional representation baselines are [ICLR 2025 B-spline KAN](https://github.com/KindXiaoming/pykan) and [CMAME 2026 RGA-KAN](https://doi.org/10.1016/j.cma.2026.118761), adapted to terminal-state regression.

## Training

```bash
python scripts/v5/train_recent.py --datasets public simulation legacy
python scripts/v5/train_recent.py --datasets public simulation legacy --methods RGA-KAN-2026 --widths 64 --glorot-sine
python scripts/v5/refit_recent_heads.py
python scripts/v4/train_models.py --dataset public --kind Geometric-KAN --coordinates observer --epochs 150
python scripts/v5/refit_primary_gdl.py
python scripts/v5/train_original_correction.py --epochs 148 --widths 64
python scripts/v5/train_original_correction.py --resume --epochs 100 --widths 64
python scripts/v5/refit_corrected_heads.py
```

Outputs are written to `runs/`. Affine-head fitting uses training data and validation selection. The historical model retains the supplied KAN encoder and refines range/velocity heads; full-test position RMSE decreases from 157.61 to 62.78 km, with median 8.96 km. The three target-disjoint cohorts retain their own frozen comparisons. Full tables also report cases where another method has lower component error.
