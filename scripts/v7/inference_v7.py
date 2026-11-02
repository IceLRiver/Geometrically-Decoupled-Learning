"""Observation-only FP64 APIs for revision candidates and frozen models."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from data_v7 import ROOT,BASE,kinematics,observer_frame
from stable_inference import load_checkpoint,predict as predict_standard,load_corrected as load_original,predict_corrected
from train_direct_velocity_heads import CorrectedHeads
from rate_heads import RateHeads
from head_design import encode
from timeformer import TimeFormer
from head_design import design

def load(spec,device='cpu'):
 path=Path(spec['checkpoint']);path=path if path.is_absolute() else BASE/path;kind=spec['kind']
 if kind in ['standard','residual_standard']:
  model,cfg=load_checkpoint(path,device);bundle=dict(kind=kind,model=model,config=cfg)
  if kind=='residual_standard':
   c=torch.load(path,map_location='cpu')['residual_heads']
   def move(v):return v.to(device=device,dtype=torch.float64) if isinstance(v,torch.Tensor) else {k:move(x) for k,x in v.items()} if isinstance(v,dict) else v
   bundle['correction']=move(c)
  return bundle
 if kind in ['corrected_original','corrected_direct','corrected_observer','corrected_observer_rate']:
  basepath=path if kind=='corrected_original' else BASE/'checkpoints/v5/legacy__GDL.pt'
  original=load_original(basepath,encoder_path=BASE/'checkpoints/v5/original_64d4.pth',norm_path=BASE/'data/legacy_simulation/norm_stats_legacy.npz',device=device)
  if kind!='corrected_original':
   c=torch.load(path,map_location='cpu');stats={k:c['state'][k] for k in ['f_mean','f_std','rho_res_std','vres_mean','vres_std']};head=(RateHeads if kind=='corrected_observer_rate' else CorrectedHeads)(stats,c['width']).to(device).double();head.load_state_dict(c['state']);head.eval();original['head']=head;original['full_observations']=c['config'].get('full_observations',False)
  return dict(kind=kind,bundle=original)
 if kind=='timeformer':
  c=torch.load(path,map_location='cpu');cfg=c['config'];m=TimeFormer(cfg['width'],cfg['scales'],cfg['gamma']).to(device).double().eval();m.load_state_dict(c['state']);return dict(kind=kind,model=m,checkpoint=c,device=device)
 raise ValueError(kind)

def predict(bundle,X,batch_size=128):
 X=np.asarray(X);kind=bundle['kind']
 if kind=='standard':return predict_standard(bundle['model'],bundle['config'],X,batch_size)
 if kind=='residual_standard':
  m=bundle['model'];cfg=bundle['config'];c=bundle['correction'];device=next(m.parameters()).device;extra=None
  if cfg['coordinates']=='hybrid':_,rawdc,_=kinematics(X);extra=np.c_[X[:,0,:3]/42000,rawdc/8]
  xx,_,B=observer_frame(X,np.zeros((len(X),6)));f,dc,dl=kinematics(xx)
  if extra is not None:f=np.c_[f,extra]
  out=[]
  with torch.no_grad():
   for j in range(0,len(X),batch_size):
    tx,tf,tdc,tdl=[torch.tensor(a[j:j+batch_size],device=device,dtype=torch.float64) for a in [xx,f,dc,dl]];hh=[];hook=m.range.register_forward_pre_hook(lambda _,args:hh.append(args[0]));_,_,baseR=m(tx,tf,tdc,tdl);hook.remove();h=hh[0];e=((tf-c['extra_mean'])/c['extra_std']).clamp(-12,12)
    if c['cross']:
     ij=torch.triu_indices(e.shape[1],e.shape[1],device=device);e=torch.cat([e,e[:,ij[0]]*e[:,ij[1]]],1)
    phi=torch.cat([design(m.range,h),e],1);rconf=c['range'];delta=((phi-rconf['mean'])/rconf['std'])@rconf['w']+rconf['bias'];r=torch.nn.functional.softplus(baseR+delta);los=tx[:,-1,3:];los=los/los.norm(dim=1,keepdim=True);p=tx[:,-1,:3]+r*los;vf=torch.cat([h,p/42000],1);prior=tdc+r*tdl+m.res_mean+m.res_std*m.velocity(vf);phi=torch.cat([design(m.velocity,vf),e],1);vconf=c['velocity'];v=prior+((phi-vconf['mean'])/vconf['std'])@vconf['w']+vconf['bias'];out.append(np.c_[p.cpu().numpy(),v.cpu().numpy()])
  z=np.concatenate(out);return np.c_[np.einsum('nbi,nb->ni',B,z[:,:3]),np.einsum('nbi,nb->ni',B,z[:,3:])]
 if kind=='corrected_original':return predict_corrected(bundle['bundle'],X,batch_size)
 if kind in ['corrected_direct','corrected_observer','corrected_observer_rate']:
  b=bundle['bundle'];device=b['device'];enc=b['encoder'];norm=b['norm'];xx=X;B=None
  if kind.startswith('corrected_observer'):xx,_,B=observer_frame(X,np.zeros((len(X),6)))
  f,dc,dl=kinematics(xx);out=[]
  with torch.no_grad():
   for j in range(0,len(X),batch_size):
    raw=(X[j:j+batch_size].reshape(-1,240).astype(float)-norm['X_mean'])/(norm['X_std']+1e-12);h=encode(enc,torch.tensor(raw,device=device,dtype=torch.float64));rho=enc.f1(h)*enc.rho_std+enc.rho_mean;parts=[h,torch.tensor(f[j:j+batch_size],device=device,dtype=torch.float64)]
    if b.get('full_observations',False):
     seq=xx[j:j+batch_size].astype(float,copy=True);seq[:,:,:3]/=42000;parts.append(torch.tensor(seq.reshape(-1,240),device=device,dtype=torch.float64))
    features=torch.cat(parts+[rho/42000],1);cam=torch.tensor(xx[j:j+batch_size,-1,:3],device=device,dtype=torch.float64);los=torch.tensor(xx[j:j+batch_size,-1,3:],device=device,dtype=torch.float64);los/=los.norm(dim=1,keepdim=True)
    p,v,_=b['head'](features,rho,cam,los,torch.tensor(dc[j:j+batch_size],device=device,dtype=torch.float64),torch.tensor(dl[j:j+batch_size],device=device,dtype=torch.float64));out.append(np.c_[p.cpu().numpy(),v.cpu().numpy()])
  z=np.concatenate(out)
  if B is not None:z=np.c_[np.einsum('nbi,nb->ni',B,z[:,:3]),np.einsum('nbi,nb->ni',B,z[:,3:])]
  return z
 if kind=='timeformer':
  c=bundle['checkpoint'];m=bundle['model'];device=bundle['device'];xx,_,B=observer_frame(X,np.zeros((len(X),6)));refit=c.get('head_refit',dict(original=True));out=[]
  with torch.no_grad():
   for j in range(0,len(X),batch_size):
    x=xx[j:j+batch_size];h=[]
    hook=m.out.register_forward_pre_hook(lambda _,args:h.append(args[0])) if not refit['original'] else None
    p=m(torch.tensor((x-c['mu'])/c['std'],device=device,dtype=torch.float64))
    if hook:hook.remove()
    if refit['original']:p=p.cpu()*torch.tensor(c['ys'])+torch.tensor(c['ym'])
    else:
     f=torch.tensor(kinematics(x)[0],dtype=torch.float64);e=((f-refit['kinematic_mean'])/refit['kinematic_std']).clamp(-12,12)
     if refit['cross']:
      ij=torch.triu_indices(e.shape[1],e.shape[1]);e=torch.cat([e,e[:,ij[0]]*e[:,ij[1]]],1)
     phi=torch.cat([h[0].cpu(),e],1);p=((phi-refit['feature_mean'])/refit['feature_std'])@refit['w']*refit['output_std']+refit['output_mean']
    out.append(p.numpy())
  z=np.concatenate(out);return np.c_[np.einsum('nbi,nb->ni',B,z[:,:3]),np.einsum('nbi,nb->ni',B,z[:,3:])]
 raise ValueError(kind)

def selected(ds,method='GDL',device='cpu'):
 specs=json.loads((BASE/'evidence/v7/models.json').read_text(encoding='utf-8'));return load(specs['datasets'][ds][method],device)
