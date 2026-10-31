# Observation arrays

`X[...,0:3]`: observer position in km. `X[...,3:6]`: unit LOS. `y[...,0:3]`: terminal target position in km. `y[...,3:6]`: terminal target velocity in km/s. Each arc has 40 one-second samples and 39 s elapsed time.

The current manuscript evaluates public precise-orbit, original simulation and real optical observations. Sentinel-1 partitions use disjoint satellite identities. Simulation retains the original sample-random partition, with shared target populations and disclosed historical checkpoint selection. Real partitions have disjoint target identities; velocity references use five-point derivatives of supplied positions. Primary indices are in `data/benchmark_protocol_v6.json`. The controlled-noise collection pairs 20 original simulation arcs at four added perturbation levels.
