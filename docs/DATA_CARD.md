# Data schema

| Field | Meaning |
|---|---|
| `X[...,0:3]` | Observer position, km |
| `X[...,3:6]` | Unit line-of-sight direction |
| `y[...,0:3]` | Terminal target position, km |
| `y[...,3:6]` | Terminal target velocity, km/s |

Inputs contain 40 time samples at one-second spacing. Original input arrays in `data/legacy_simulation` can be reshaped to `(N,40,6)`. Current benchmark arrays already use that shape. `split` codes 0/1/2 indicate training/validation/test. Frozen primary evaluation indices are in `data/benchmark_protocol_v4.json`; each method uses the same 300 public and 300 simulation arcs. Satellite identities separate the Sentinel-1 partitions, and orbit identities separate the multiregime partitions. The latter has 40,000 arcs per orbital regime and noise levels 0/1/3/5 arcsec. Sentinel-1 LOS observations are reconstructed with a 3-arcsec tangent-plane noise level and are not sensor measurements.
