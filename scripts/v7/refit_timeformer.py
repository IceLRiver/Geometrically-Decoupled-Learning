"""Validation-selected affine state head; observations/kinematics only."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from data_v7 import ROOT,load_data,observer_frame,kinematics
from timeformer import TimeFormer
torch.set_num_threads(3)
OUT=ROOT/'evidence/timeformer_refitted';OUT.mkdir(exist_ok=True)
for ds in ['public','legacy']:
 X,y,_,idx=load_data(ds);truth=np.asarray(y,float);X,y,B=observer_frame(X,y);ids=np.r_[idx['train'],idx['val']];n=len(idx['train']);X=X[ids];Y=torch.tensor(y[ids],dtype=torch.float64);f=torch.tensor(kinematics(X)[0],dtype=torch.float64);sc=[np.linalg.norm(truth[idx['train'],:3].std(0)),np.linalg.norm(truth[idx['train'],3:].std(0))];allruns=[]
 for path in sorted((ROOT/'evidence/timeformer_formal').glob(ds+'_*.pt')):
  existing=OUT/path.name
  if existing.exists():
   saved=torch.load(existing,map_location='cpu');cfg=saved['config'];refit=saved['head_refit'];count=saved['parameters']+(0 if refit['original'] else refit['w'].numel()-6*(6*cfg['scales']*cfg['width']));allruns.append(dict(checkpoint=str(existing.relative_to(ROOT)),validation=refit['validation'],parameters=count));print('REUSED',ds,path.name,refit['validation'],flush=True);continue
  c=torch.load(path,map_location='cpu');cfg=c['config'];m=TimeFormer(cfg['width'],cfg['scales'],cfg['gamma']).cuda().double().eval();m.load_state_dict(c['state']);h=[];hook=m.out.register_forward_pre_hook(lambda _,args:h.append(args[0].detach().cpu()));base=[]
  with torch.no_grad():
   for j in range(0,len(X),256):base.append(m(torch.tensor((X[j:j+256]-c['mu'])/c['std'],device='cuda',dtype=torch.float64)).cpu())
  hook.remove();h=torch.cat(h);base=torch.cat(base)*torch.tensor(c['ys'])+torch.tensor(c['ym']);mu=f[:n].mean(0);std=f[:n].std(0).clamp_min(1e-7);e=((f-mu)/std).clamp(-12,12);selected=None;records=[]
  def score(p):
   d=p[n:]-Y[n:];er=d[:,:3].square().sum(1).mean().sqrt().item();ev=d[:,3:].square().sum(1).mean().sqrt().item();return dict(position_km_rmse=er,velocity_km_s_rmse=ev,common_score=er/sc[0]+ev/sc[1])
  best=score(base);selected=dict(original=True,validation=best)
  for cross in [False,True]:
   extra=e
   if cross:
    ij=torch.triu_indices(e.shape[1],e.shape[1]);extra=torch.cat([e,e[:,ij[0]]*e[:,ij[1]]],1)
   phi=torch.cat([h,extra],1);pm=phi[:n].mean(0);ps=phi[:n].std(0).clamp_min(1e-7);z=(phi-pm)/ps;ym=Y[:n].mean(0);ys=Y[:n].std(0).clamp_min(1e-3);target=(Y[:n]-ym)/ys;A=z[:n].T@z[:n]/n;b=z[:n].T@target/n
   for ridge in [1e-10,1e-8,1e-6,.0001,.01,1.]:
    w=torch.linalg.solve(A+ridge*torch.eye(A.shape[0],dtype=torch.float64),b);metrics=score((z@w)*ys+ym);row=dict(cross=cross,ridge=ridge,**metrics);records.append(row)
    if metrics['common_score']<selected['validation']['common_score']:selected=dict(original=False,validation=metrics,cross=cross,w=w,feature_mean=pm,feature_std=ps,kinematic_mean=mu,kinematic_std=std,output_mean=ym,output_std=ys)
  c['head_refit']=selected;dest=OUT/path.name;torch.save(c,dest);allruns.append(dict(checkpoint=str(dest.relative_to(ROOT)),validation=selected['validation'],parameters=sum(p.numel() for p in m.parameters())+(0 if selected['original'] else selected['w'].numel()+6-m.out.weight.numel()-m.out.bias.numel())));(OUT/(path.stem+'.json')).write_text(json.dumps(dict(validation=selected['validation'],original=selected['original'],candidates=records),indent=2));print(ds,path.name,selected['validation'],flush=True)
 selected=dict(min(allruns,key=lambda r:r['validation']['common_score']));selected['sha256']=hashlib.sha256((ROOT/selected['checkpoint']).read_bytes()).hexdigest();selected['candidates']=allruns;(OUT/(ds+'_selected.json')).write_text(json.dumps(selected,indent=2))
print('TIMEFORMER CALIBRATION COMPLETE',flush=True)
