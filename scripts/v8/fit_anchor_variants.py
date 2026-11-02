from pathlib import Path
import json
import numpy as np
import torch
from duration_features import R,velocity_design
from native_derivatives import derivatives
ns={};src=(R/'fit_duration_heads.py').read_text();exec(src[:src.index("for ds in ['public','legacy']:")],ns);ns['RIDGES']=[1e-6,1e-4,.001,.003,.01,.03,.1,.3,1.,100.];extras,fits,apply,errors=[ns[k] for k in ['extras','fits','apply','errors']]
E=R/'evidence/duration';O=E/'adapted'
for ds in ['public']:
 from native_data import load_native
 z=load_native(ds);Yall=z['y'];tr=z['train'];sc=[np.linalg.norm(Yall[tr,:3].std(0)),np.linalg.norm(Yall[tr,3:].std(0))]
 for N in [10,20,60]:
  dest=O/f'{ds}_{N}_GDL_anchor.pt'
  if dest.exists():continue
  base=torch.load(O/f'{ds}_{N}_GDL.pt',map_location='cpu');c=np.load(E/'features'/f'{ds}_{N}_GDL.npz');n=int(c['n_train']);ids=c['indices'];native=z['X60'][ids,-N:];Y=np.c_[np.einsum('nbi,ni->nb',c['B'],Yall[ids,:3]),np.einsum('nbi,ni->nb',c['B'],Yall[ids,3:])];ex,exstats=extras(c,base['cross'],base['extra_stats']);r=np.logaddexp(0,apply(base['range'],np.c_[c['phi'],ex]));p=c['cam']+r*c['los'];vp0=velocity_design(ds,c['h'],p);best=None;records=[]
  for degree in [2,3,4]:
   dc,dl=derivatives(native,c['B'],degree);anchor=dc+r*dl;vp=np.c_[vp0,ex,dc/8,dl/.001]
   for rate in [False,True]:
    target=((Y[:,3:]-anchor)*c['los']).sum(1,keepdims=True) if rate else Y[:,3:]-anchor
    for residual,vc in fits(vp,target,n):
     v=anchor+residual*c['los'] if rate else anchor+residual;val=errors(p,v,Y,n,sc);records.append(dict(degree=degree,range_rate=rate,ridge=vc['ridge'],**val))
     if best is None or val['common_score']<best['validation']['common_score']:best=dict(base,velocity=vc,range_rate=rate,anchor_degree=degree,append_anchor_features=True,validation=val)
  torch.save(best,dest);dest.with_suffix('.json').write_text(json.dumps(dict(validation=best['validation'],candidates=records),indent=2));print(dest.stem,best['validation'],'degree',best['anchor_degree'],'rate',best['range_rate'],flush=True)
print('ANCHOR VALIDATION COMPLETE',flush=True)
