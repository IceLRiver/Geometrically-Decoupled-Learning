from paths import BASE
import argparse,json,numpy as np
from physics_stable import solve
from rbf_initializer import rbf
from data_v8 import load_data,protocol
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['public','legacy'],default='public');p.add_argument('--method',choices=['PIOD-2023','alpha-IOD-2025','PIOD-2026'],default='PIOD-2023');p.add_argument('--limit',type=int,default=1);a=p.parse_args()
f=json.loads((BASE/'evidence/v8/models.json').read_text());config=dict(f['physics'][a.dataset][a.method]['config']);mode=config.pop('initialization');prior=json.loads((BASE/'evidence/v8/priors.json').read_text())[a.dataset];X,y,_,_=load_data(a.dataset);ids=np.asarray(protocol()['datasets'][a.dataset]['test_primary'])[:a.limit];initial=rbf(a.dataset,X[ids]) if mode=='RBFNN' else None;saved=np.load(BASE/'evidence/v8/predictions'/f'{a.dataset}__{a.method}.npz')['state']
for j,i in enumerate(ids):
 state,detail=solve(X[i],a.method,seed=42+int(i),prior=prior,init_state=initial[j] if initial is not None else None,runs=100,**config);assert np.allclose(state,saved[j],atol=2e-5,rtol=1e-8);print(int(i),'saved state reproduced',detail,flush=True)
