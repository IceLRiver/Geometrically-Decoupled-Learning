# Observation arrays

`X[...,0:3]`: observer position in km. `X[...,3:6]`: unit LOS. `y[...,0:3]`: terminal target position in km. `y[...,3:6]`: terminal target velocity in km/s. Each arc has 40 one-second samples and 39 s elapsed time.

Sentinel-1 partitions use disjoint satellite identities. Multiregime partitions use disjoint orbit identities, equal LEO/MEO/high-altitude populations and noise codes 0/1/3/5 arcsec. Original sample-random partitions retain shared target populations; the supplied checkpoint's historical selection is disclosed. Primary test indices are stored with the arrays. The controlled measurement set pairs the same 20 original arcs across four added noise levels. State errors for these distinct protocols are reported separately.
