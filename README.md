# Geometrically Decoupled Learning

Data and model code for **Geometrically Decoupled Learning for Space Target Motion State Estimation from Short-Arc Angles-Only Observations**.

## Data

| Dataset | Arcs | Train / validation / test |
|---|---:|---|
| Sentinel-1 precise-orbit benchmark | 9,632 | 4,815 / 2,412 / 2,405; satellite-disjoint |
| Simulation | 236,575 | 170,334 / 18,926 / 47,315; original sample split |
| Real optical observations | 818 | 600 / 96 / 122; target-disjoint, access controlled |

Each input `X` has shape `(N,40,6)`: observer position in km and unit LOS at one-second cadence. Labels `y` contain terminal position (km) and velocity (km/s). Simulation arrays in `data/legacy_simulation` are stored as `(N,240)` and are selected with the `legacy` dataset argument. Sentinel-1 uses reconstructed angles with precise ephemeris references. Contains modified Copernicus Sentinel data (2021, 2026); [source catalogue](https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/).

`data/benchmark_protocol_v6.json` records the current common 300 public, 300 simulation and 122 real comparison indices. The original simulation checkpoint and normalization have a historical selection history; the simulation study is retrospective. `data/controlled_angle_noise` pairs 20 simulation arcs over four additional noise levels. Real sensor records remain subject to the provider's access conditions.

## Installation and evaluation

Use Python 3.9¨C3.11 for neural models. Numerical orbit/RBF solvers use a separate Python 3.11 environment with `requirements-orbits.txt`.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/v6/verify_components.py
python scripts/v5/evaluate_checkpoint.py --dataset public --method GDL
python scripts/v5/evaluate_checkpoint.py --dataset legacy --method GDL --full-test
python scripts/v5/evaluate_checkpoint.py --dataset legacy --method B-spline-KAN-2025
python scripts/v5/evaluate_checkpoint.py --dataset public --method RGA-KAN-2026
python scripts/v5/evaluate_physics.py --dataset legacy --method XGEO-PIOD --workers 4
```

`models/geometric_v5.py` exposes observation-only `load_model(checkpoint_path)` and `predict_states(bundle, X)`. Evaluation scripts use `--dataset legacy` for the retained simulation benchmark. Training uses FP32; final inference uses FP64 to stabilize calibrated basis sums. Current aggregate results and the rerun component study are in `evidence/v6/`; the unchanged primary frozen weights and full-test predictions are in `checkpoints/v5/` and `evidence/v5/`.

All ten primary methods share identical comparison arcs. Table entries identify PIOD (2023), RBFNN (2024), ¦Á-IOD (2025), XGEO-PIOD (2026), KAN (2025), RGA-KAN (2026), and the present GDL study (2026). Classical comparisons are Gauss, Laplace and Gooding-type. Physics-based rows are documented task reimplementations. [B-spline KAN](https://github.com/KindXiaoming/pykan) and [RGA-KAN](https://doi.org/10.1016/j.cma.2026.118761) are adapted to supervised terminal-state regression.

## Training

```bash
python scripts/v5/train_recent.py --datasets public legacy
python scripts/v5/train_recent.py --datasets public legacy --methods RGA-KAN-2026 --widths 64 --glorot-sine
python scripts/v5/refit_recent_heads.py
python scripts/v4/train_models.py --dataset public --kind Geometric-KAN --coordinates observer --epochs 150
python scripts/v5/train_original_correction.py --epochs 148 --widths 64
python scripts/v5/train_original_correction.py --resume --epochs 100 --widths 64
python scripts/v5/refit_corrected_heads.py
python scripts/v6/train_simulation_ablation.py
```

Outputs are written to `runs/`. The primary simulation model retains a supplied four-block encoder and refines geometric heads: full-test position RMSE is 62.78 km, with median 8.96 km. The controlled component study trains smaller width-32, one-block variants from scratch without head calibration. Its errors and capacity counts therefore describe a different training budget, rather than the pretrained primary model. The directional-density and parameter-efficiency analyses use saved predictions from the current three datasets.
