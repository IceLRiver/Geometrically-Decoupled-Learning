from pathlib import Path
import sys,json
import numpy as np
import torch
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'scripts/v7'))
from data_v7 import load_data
from inference_v7 import selected,predict
torch.set_num_threads(4);records=[]
for ds in ['public','legacy']:
 X,_,_,_=load_data(ds)
 for method in ['GDL','TimeFormer-2026']:
  b=selected(ds,method,device='cpu');z=np.load(R/'evidence/v7/predictions'/(ds+'__'+method+'.npz'));ids=z['indices'][:8];p=predict(b,X[ids]);single=predict(b,X[ids[:1]]);e=np.abs(p-z['state'][:8]);tol=2e-4 if method=='TimeFormer-2026' else 1e-5;assert e[:,:3].max()<tol and e[:,3:].max()<tol/100,(ds,method,e.max());assert np.abs(p[:1]-single).max()<tol
  records.append(dict(dataset=ds,method=method,max_position_difference_km=float(e[:,:3].max()),max_velocity_difference_km_s=float(e[:,3:].max())));print(ds,method,'API PASS',flush=True)
(R/'evidence/v7/public_api_checks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
