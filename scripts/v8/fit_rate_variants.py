"""Validation-only scalar range-rate alternatives; fixed feature representations."""
from pathlib import Path
import json
import numpy as np
import torch
from duration_features import R,velocity_design
# Import definitions without executing the batch fitter.
src=(R/'fit_duration_heads.py').read_text();ns={};exec(src[:src.index("for ds in ['public','legacy']:")],ns)
extras,fits,apply,errors=[ns[k] for k in ['extras','fits','apply','errors']]
E=R/'evidence/duration';O=E/'adapted'
for ds in ['public']:
 from native_data import load_native
 z=load_native(ds);Yall=z['y'];tr=z['train'];sc=[np.linalg.norm(Yall[tr,:3].std(0)),np.linalg.norm(Yall[tr,3:].std(0))]
 for N in [10,20,60]:
  dest=O/f'{ds}_{N}_GDL_rate.pt'
  if dest.exists():continue
  c=np.load(E/'features'/f'{ds}_{N}_GDL.npz');n=int(c['n_train']);ids=c['indices'];Y=np.c_[np.einsum('nbi,ni->nb',c['B'],Yall[ids,:3]),np.einsum('nbi,ni->nb',c['B'],Yall[ids,3:])];best=None;records=[]
  for cross in [False,True]:
   ex,stats=extras(c,cross,n=n);phi=np.c_[c['phi'],ex];rho=np.linalg.norm(Y[:,:3]-c['cam'],axis=1,keepdims=True);bestR=None
   for r,rc in fits(phi,rho,n):
    r=np.logaddexp(0,r);p=c['cam']+r*c['los'];er=np.sqrt(np.mean(np.sum((p[n:]-Y[n:,:3])**2,1)))
    if bestR is None or er<bestR[0]:bestR=(er,r,rc,p)
   _,r,rc,p=bestR;vp=np.c_[velocity_design(ds,c['h'],p),ex];anchor=c['dc']+r*c['dl'];target=((Y[:,3:]-anchor)*c['los']).sum(1,keepdims=True)
   for residual,vc in fits(vp,target,n):
    v=anchor+residual*c['los'];val=errors(p,v,Y,n,sc);records.append(dict(cross=cross,range_ridge=rc['ridge'],velocity_ridge=vc['ridge'],**val))
    if best is None or val['common_score']<best['validation']['common_score']:best=dict(dataset=ds,method='GDL',measurements=N,cross=cross,extra_stats=stats,range=rc,velocity=vc,range_rate=True,validation=val,training_rows=n,validation_rows=len(ids)-n)
  torch.save(best,dest);dest.with_suffix('.json').write_text(json.dumps(dict(validation=best['validation'],candidates=records),indent=2));print(dest.stem,best['validation'],flush=True)
print('RANGE-RATE VALIDATION COMPLETE',flush=True)
