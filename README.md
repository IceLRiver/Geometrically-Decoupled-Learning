# Geometrically Decoupled Learning

Data and model code for **Geometrically Decoupled Learning for Space Target Motion State Estimation from Short-Arc Angles-Only Observations**.

## Data

| Dataset | Arcs | Train / validation / test |
|---|---:|---|
| Sentinel-1 precise-orbit benchmark | 9,632 | 4,815 / 2,412 / 2,405; satellite-disjoint |
| Simulation | 236,575 | 170,334 / 18,926 / 47,315; original sample split |
| Real optical observations | 818 | 600 / 96 / 122; target-disjoint, access controlled |

Inputs contain observer positions (km) and unit LOS, with shape `(N,40,6)` and one-second cadence. Labels contain terminal position (km) and velocity (km/s). Simulation chunks in `data/legacy_simulation` use flattened `(N,240)` observations; the dataset argument is `legacy`. Sentinel-1 observations are reconstructed angles with precise ephemeris references. Contains modified Copernicus Sentinel data (2021, 2026); [source catalogue](https://step.esa.int/auxdata/orbits/Sentinel-1/POEORB/). Private sensor records remain subject to provider access conditions.

The current common comparison has 300 public, 300 simulation and 122 real arcs, recorded in `data/benchmark_protocol_v7.json`. The simulation evaluation retains the original sample split and historically selected source encoder/normalization. It is a retrospective evaluation on a potentially shared target population.

## Evaluate the current models

```bash
python -m pip install -r requirements.txt
python scripts/v7/verify_results.py
python scripts/v7/check_api.py
python scripts/v7/evaluate_checkpoint.py --dataset public --method GDL
python scripts/v7/evaluate_checkpoint.py --dataset legacy --method TimeFormer-2026
```

`models/geometric_v7.py` provides `load_model(dataset, method, device)` and `predict_states(bundle, X)`. Prediction accepts observations only. Current checkpoints are in `checkpoints/v7`; aggregate metrics and public/simulation predictions are in `evidence/v7`. Earlier comparison weights remain unchanged.

| Method | Public position / velocity RMSE | Simulation position / velocity RMSE | Real position / velocity RMSE |
|---|---:|---:|---:|
| GDL | 0.424 km / 0.0197 km/s | 23.08 km / 0.0165 km/s | 119.52 km / 0.0107 km/s |
| TimeFormer (2026), state adaptation | 12.66 km / 0.3556 km/s | 242.50 km / 0.1230 km/s | 145.35 km / 0.0153 km/s |

Eleven primary methods share identical cohorts. Literature methods are PIOD (2023), RBFNN (2024), α-IOD (2025), KAN (2025), XGEO-PIOD (2026), RGA-KAN (2026) and [TimeFormer (2026)](https://www.sciencedirect.com/science/article/pii/S0957417425046548), plus GDL and three classical solvers. KAN and TimeFormer comparisons are documented six-state regression adaptations. `docs/METHODS_V7.md` describes the adaptation and training settings.

## Training

Current training scripts consume the released observation arrays. Outputs go to `runs/v7` (override `ORBIT_RUN_ROOT` to choose another directory). The simulation model uses the supplied four-block frozen encoder and width-128 range/range-rate heads. Public/real GDL use their geometric backbone with training-only residual-head fitting. Final inference uses FP64.

```bash
python scripts/v7/cache_observer_features.py
python scripts/v7/train_observer_heads.py
python scripts/v7/train_observer_heads.py --full-observations
python scripts/v7/tune_velocity_balance.py
python scripts/v7/tune_velocity_balance.py --stronger
python scripts/v7/refit_observer_heads.py
python scripts/v7/tune_residual_heads.py
python scripts/v7/train_timeformer.py
python scripts/v7/refit_timeformer.py
```

The six width-32 from-scratch component variants retain their previous common optimizer budget; their artifacts are in `evidence/v6` and `checkpoints/v6`. The full-model reference uses a larger training/capacity budget and is identified separately. Parameter--error comparisons use only the simulation cohort and include all primary methods plus the component variants.

## Observation duration

The fixed-model study compares 9, 19, 29 and 39 seconds on 20 matched arcs per source. Classical/physics solvers use native measured samples. Fixed-size learned inputs interpolate the shorter measured subarc to 40 computational entries and use its physical cadence; interpolation adds no independent measurements. All original 39-second comparison predictions are preserved. GDL degrades markedly below its 39-second training span, so the duration results do not establish robustness to arbitrary input lengths. Full values are in `evidence/v7/arc_length.csv`.
