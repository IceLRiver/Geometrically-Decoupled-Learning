"""Preserve the fitted GDL map, fit observation-only residual head coefficients.
All coefficient fitting uses training rows; all choices use validation rows.
"""
from pathlib import Path
import json,copy
import numpy as np
import torch
from data_v7 import ROOT,BASE,load_data,observer_frame,kinematics
from stable_inference import load_checkpoint
from head_design import design
torch.set_num_threads(3)
OUT=ROOT/'evidence/residual_heads';OUT.mkdir(exist_ok=True)
def regress(phi,target,n):
 mu=phi[:n].mean(0);std=phi[:n].std(0).clamp_min(1e-7);z=(phi-mu)/std;tm=target[:n].mean(0);A=z[:n].T@z[:n]/n;b=z[:n].T@(target[:n]-tm)/n
 for ridge in [1e-8,1e-6,.0001,.001,.01,.1,1.,100.]:
  w=torch.linalg.solve(A+ridge*torch.eye(A.shape[0],dtype=torch.float64),b);yield dict(w=w,mean=mu,std=std,bias=tm,ridge=ridge),z@w+tm
for ds in ['public']:
 raw,y,_,idx=load_data(ds);tr=idx['train'];va=idx['val'];ids=np.r_[tr,va];n=len(tr);sc=[np.linalg.norm(y[tr,:3].astype(float).std(0)),np.linalg.norm(y[tr,3:].astype(float).std(0))];_,rawdc,_=kinematics(raw);hybrid=np.c_[raw[:,0,:3]/42000,rawdc/8];X,y,B=observer_frame(raw,y);X=X[ids];Y=torch.tensor(y[ids],dtype=torch.float64);f,dc,dl=kinematics(X);basepath=BASE/'checkpoints/v5'/f'{ds}__GDL.pt';m,cfg=load_checkpoint(basepath,device='cpu')
 if cfg['coordinates']=='hybrid':f=np.c_[f,hybrid[ids]]
 F=torch.tensor(f,dtype=torch.float64);DC=torch.tensor(dc,dtype=torch.float64);DL=torch.tensor(dl,dtype=torch.float64);CAM=torch.tensor(X[:,-1,:3]);LOS=torch.tensor(X[:,-1,3:]);LOS=LOS/LOS.norm(dim=1,keepdim=True);H=[];ps=[];vs=[];rs=[];hook=m.range.register_forward_pre_hook(lambda _,args:H.append(args[0].detach()))
 with torch.no_grad():
  for j in range(0,len(ids),128):
   p,v,r=m(torch.tensor(X[j:j+128],dtype=torch.float64),F[j:j+128],DC[j:j+128],DL[j:j+128]);ps.append(p);vs.append(v);rs.append(r)
 hook.remove();h=torch.cat(H);baseR=torch.cat(rs);trueR=(Y[:,:3]-CAM).norm(dim=1,keepdim=True)
 def errors(p,v):
  d=p[n:]-Y[n:,:3];dv=v[n:]-Y[n:,3:];er=d.square().sum(1).mean().sqrt().item();ev=dv.square().sum(1).mean().sqrt().item();return dict(position_km_rmse=er,velocity_km_s_rmse=ev,common_score=er/sc[0]+ev/sc[1])
 best=errors(torch.cat(ps),torch.cat(vs));selected=dict(original=True,validation=best);records=[]
 with torch.no_grad():
  em=F[:n].mean(0);es=F[:n].std(0).clamp_min(1e-7);e=((F-em)/es).clamp(-12,12)
  for cross in [False,True]:
   extra=e
   if cross:
    ij=torch.triu_indices(e.shape[1],e.shape[1]);extra=torch.cat([e,e[:,ij[0]]*e[:,ij[1]]],1)
   rp=torch.cat([design(m.range,h),extra],1);rcandidates=list(regress(rp,trueR-baseR,n));zero=dict(w=torch.zeros((rp.shape[1],1),dtype=torch.float64),mean=rp[:n].mean(0),std=rp[:n].std(0).clamp_min(1e-7),bias=torch.zeros(1,dtype=torch.float64),ridge=None);rcandidates.append((zero,torch.zeros_like(baseR)))
   for rc,deltaR in rcandidates:
    r=torch.nn.functional.softplus(baseR+deltaR);p=CAM+r*LOS;vf=torch.cat([h,p/42000],1);prior=DC+r*DL+m.res_mean+m.res_std*m.velocity(vf);vp=torch.cat([design(m.velocity,vf),extra],1)
    for vc,deltaV in regress(vp,Y[:,3:]-prior,n):
     val=errors(p,prior+deltaV);records.append(dict(cross=cross,range_ridge=rc['ridge'],velocity_ridge=vc['ridge'],**val))
     if val['common_score']<selected['validation']['common_score']-max(1e-9,1e-5*selected['validation']['common_score']):
      selected=dict(original=False,validation=val,cross=cross,range=rc,velocity=vc,extra_mean=em,extra_std=es);ck=torch.load(basepath,map_location='cpu');ck['residual_heads']=selected;torch.save(ck,OUT/(ds+'.pt'))
 (OUT/(ds+'_selection.json')).write_text(json.dumps(dict(selected={k:v for k,v in selected.items() if k in ['original','validation','cross']},candidates=records),indent=2));print(ds,{k:v for k,v in selected.items() if k in ['original','validation','cross']},flush=True)
print('RESIDUAL VALIDATION SEARCH COMPLETE',flush=True)
