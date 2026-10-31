from pathlib import Path
import sys,json
import numpy as np
BASE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(BASE/'scripts/v5'))
from data_v5 import load_data
rows=json.loads((BASE/'evidence/v6/component_results.json').read_text());count=0
for ds in ['public','legacy']:
    _,y,_,idx=load_data(ds)
    for r in [r for r in rows if r['dataset']==ds]:
        with np.load(BASE/'evidence/v6/predictions'/f"{ds}_{r['kind']}.npz") as z:
            ids=z['indices'];assert set(ids)<=set(idx['test']);d=z['state'].astype(float)-y[ids].astype(float)
            value=[float(np.sqrt(np.mean(np.sum(d[:,k:k+3]**2,axis=1)))) for k in [0,3]]
            assert np.allclose(value,[r['position_km_rmse'],r['velocity_km_s_rmse']],rtol=1e-10,atol=1e-9),(ds,r['kind'])
            count+=1
print(f'PASS: {count} retained public/simulation component prediction rows independently rescored.')
