"""Common training-only affine-output refinement for both recent baselines.

The spline/Chebyshev output layers are linear in their basis coefficients.
Ridge strength is selected on validation states, never on test states.
"""
import json,copy,time
from pathlib import Path
import numpy as np
import torch
from data_v5 import ROOT,load_data,observer_frame
from recent_models import BSplineKAN,RGAKAN,chebyshev
from train_recent import features
torch.set_num_threads(4)

def run(path):
 c=torch.load(path,map_location='cpu');ds=c['config']['dataset'];X,y,_,idx=load_data(ds);truth=np.asarray(y,float);X,y,B=observer_frame(X,y);tr=idx['train'];va=idx['val'];ids=np.r_[tr,va];n=len(tr);f=features(X[ids]);f=np.clip((f-c['mu'])/c['std'],-6,6);model=(BSplineKAN(f.shape[1],6,c['width'],5) if c['method']=='B-spline-KAN-2025' else RGAKAN(f.shape[1],6,c['width'],5,2,c['config'].get('glorot_sine',False))).cuda().eval();model.load_state_dict(c['state']);basis=[]
 with torch.no_grad():
  for j in range(0,len(ids),1024):
   x=torch.tensor(f[j:j+1024],device='cuda')
   if c['method']=='B-spline-KAN-2025':
    x=model.layers[0](x);layer=model.layers[1];phi=torch.cat([torch.nn.functional.silu(x),layer.basis(x).flatten(1)],1)
   else:
    x=model.embed(x);u=model.U(x);v=model.V(x)
    for block in model.blocks:x=block(x,u,v)
    phi=chebyshev(x,5).flatten(1)
   basis.append(phi.cpu())
 phi=torch.cat(basis).double();mu=phi[:n].mean(0);std=phi[:n].std(0).clamp_min(1e-7);z=(phi[:n]-mu)/std;zv=(phi[n:]-mu)/std;target=torch.tensor((y[tr]-c['ym'])/c['ys'],dtype=torch.float64);tm=target.mean(0);A=z.T@z/n;b=z.T@(target-tm)/n;scales=[np.linalg.norm(truth[tr,:3].std(0)),np.linalg.norm(truth[tr,3:].std(0))];records=[];best=None
 for ridge in [1e-10,1e-8,1e-6,1e-4,.01,1.]:
  w=torch.linalg.solve(A+ridge*torch.eye(A.shape[0],dtype=torch.float64),b);local=(zv@w+tm).numpy()*c['ys']+c['ym'];pred=np.c_[np.einsum('nbi,nb->ni',B[va],local[:,:3]),np.einsum('nbi,nb->ni',B[va],local[:,3:])];er=np.sqrt(np.mean(np.sum((pred[:,:3]-truth[va,:3])**2,axis=1)));ev=np.sqrt(np.mean(np.sum((pred[:,3:]-truth[va,3:])**2,axis=1)));score=er/scales[0]+ev/scales[1];row=dict(ridge=ridge,position_km_rmse=float(er),velocity_km_s_rmse=float(ev),common_score=float(score));records.append(row)
  if best is None or score<best[0]:best=(score,w/std[:,None],tm-(mu/std)@w,row)
 weight=best[1].T.float().cuda();bias=best[2].float().cuda()
 with torch.no_grad():
  if c['method']=='B-spline-KAN-2025':
   # Store the fitted affine intercept in the output de-normalization offset.
   layer=model.layers[1];layer.base.copy_(weight[:,:c['width']]);layer.coef.copy_(weight[:,c['width']:].reshape_as(layer.coef));c['original_target_mean']=c['ym'].copy();c['ym']=c['ym']+bias.cpu().numpy()*c['ys']
  else:model.output.coef.copy_(weight.reshape_as(model.output.coef));model.output.bias.copy_(bias)
 c['state']=model.state_dict();c['validation_refit']=best[3];c['training_recipe']='AdamW plus training-only affine output ridge refit';out=ROOT/'evidence/recent_refitted';out.mkdir(exist_ok=True);dest=out/path.name;torch.save(c,dest);record=dict(checkpoint=dest.relative_to(ROOT).as_posix(),dataset=ds,method=c['method'],validation=best[3],search=records);(out/(path.stem+'.json')).write_text(json.dumps(record,indent=2));print(record['checkpoint'],best[3],flush=True)
if __name__=='__main__':
 folder=ROOT/'evidence/recent_models'
 for ds in ['public','real','legacy','simulation']:
  for method in ['B-spline-KAN-2025','RGA-KAN-2026']:
   for suffix in ['','_glorot'] if method=='RGA-KAN-2026' else ['']:
    selected=folder/f'{ds}__{method}{suffix}_selected.json'
    if not selected.exists():continue
    spec=json.loads(selected.read_text());p=ROOT/spec['checkpoint']
    if not (ROOT/'evidence/recent_refitted'/p.name).exists():run(p)
