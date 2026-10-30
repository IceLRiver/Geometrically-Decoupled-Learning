# Data schema

| Field | Meaning |
|---|---|
| `X[...,0:3]` | Observer position, km |
| `X[...,3:6]` | Unit line-of-sight direction |
| `y[...,0:3]` | Terminal target position, km |
| `y[...,3:6]` | Terminal target velocity, km/s |

Inputs contain 40 time samples. Legacy input arrays are flattened and can be reshaped to `(N,40,6)`. The compact evaluation partition uses disjoint target orbits and is specified in `evidence/synthetic_split.json`.
