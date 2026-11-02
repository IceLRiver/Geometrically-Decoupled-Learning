from pathlib import Path
import sys,json
import numpy as np
import torch
from paths import R,B,BASE
import benchmark_data as bd
from inference_v7 import selected
from stable_inference import load_recent
from timed_features import cadence
from head_design import encode,design
from train_recent import features
from recent_models import chebyshev
from data_v5 import load_data
torch.set_num_threads(4)

def sample40(X,N):
 if N==40:return np.asarray(X[:,-40:],dtype=float).copy()
 native=X[:,-N:];times=np.linspace(0,N-1,40);lo=np.floor(times).astype(int);hi=np.minimum(lo+1,N-1);a=times-lo;out=native[:,lo]*(1-a[None,:,None])+native[:,hi]*a[None,:,None];out[:,:,3:]/=np.linalg.norm(out[:,:,3:],axis=2,keepdims=True);return out

def foundation(ds,method,device='cuda'):
 if method in ['GDL','TimeFormer-2026']:return selected(ds,method,device)
 return load_recent(BASE/'checkpoints/v5'/f'{ds}__{method}.pt',device)

def extract(ds,method,X,N,batch=512,device='cpu'):
 """Frozen representations only. No target/reference states enter this function."""
 bundle=foundation(ds,method,device);xx=sample40(X,N);dt=(N-1)/39
 with cadence(dt):
  xx,_,basis=bd.observer_frame(xx,np.zeros((len(xx),6)));f,dc,dl=bd.kinematics(xx)
  # Hybrid real backbone also retains its original inertial observed features.
  original=sample40(X,N);_,rawdc,_=bd.kinematics(original)
  extra=np.c_[original[:,0,:3]/42000,rawdc/8]
  ef=f if not(method=='GDL' and ds=='real') else np.c_[f,extra]
  hs=[];phis=[]
  with torch.no_grad():
   for j in range(0,len(xx),batch):
    sl=slice(j,j+batch);tx=torch.tensor(xx[sl],device=device,dtype=torch.float64)
    if method=='GDL':
     if ds=='legacy':
      b=bundle['bundle'];enc=b['encoder'];norm=b['norm'];raw=(original[sl].reshape(-1,240)-norm['X_mean'])/(norm['X_std']+1e-12);h=encode(enc,torch.tensor(raw,device=device,dtype=torch.float64));rho=enc.f1(h)*enc.rho_std+enc.rho_mean;z=xx[sl].copy();z[:,:,:3]/=42000;F=torch.cat([h,torch.tensor(f[sl],device=device,dtype=torch.float64),torch.tensor(z.reshape(-1,240),device=device,dtype=torch.float64),rho/42000],1);m=b['head'];h=m.embed(((F-m.f_mean)/m.f_std).clamp(-12,12))
     else:
      m=bundle['model'];hh=[];hook=m.range.register_forward_pre_hook(lambda _,a:hh.append(a[0]));m(tx,torch.tensor(ef[sl],device=device,dtype=torch.float64),torch.tensor(dc[sl],device=device,dtype=torch.float64),torch.tensor(dl[sl],device=device,dtype=torch.float64));hook.remove();h=hh[0]
     phi=design(m.range,h)
    elif method=='TimeFormer-2026':
     m=bundle['model'];c=bundle['checkpoint'];hh=[];hook=m.out.register_forward_pre_hook(lambda _,a:hh.append(a[0]));m(torch.tensor((xx[sl]-c['mu'])/c['std'],device=device,dtype=torch.float64));hook.remove();h=hh[0];phi=h
    else:
     m=bundle['model'];c=bundle['checkpoint'];layer=m.layers[-1] if method=='B-spline-KAN-2025' else m.output;hh=[];hook=layer.register_forward_pre_hook(lambda _,a:hh.append(a[0]));z=np.clip((features(xx[sl])-c['mu'])/c['std'],-6,6);m(torch.tensor(z,device=device,dtype=torch.float64));hook.remove();h=hh[0];phi=torch.cat([torch.nn.functional.silu(h),layer.basis(h).flatten(1)],1) if method=='B-spline-KAN-2025' else chebyshev(h,5).flatten(1)
    hs.append(h.cpu().numpy());phis.append(phi.cpu().numpy())
 h=np.concatenate(hs);phi=np.concatenate(phis);del bundle;torch.cuda.empty_cache();raw=xx.copy();raw[:,:,:3]/=42000
 return dict(h=h,phi=phi,f=f.astype(float),raw=raw.reshape(-1,240),cam=xx[:,-1,:3],los=xx[:,-1,3:],dc=dc.astype(float),dl=dl.astype(float),B=basis)

def velocity_design(ds,h,p):
 b=foundation(ds,'GDL','cpu');m=b['bundle']['head'] if ds=='legacy' else b['model'];out=[]
 with torch.no_grad():
  for j in range(0,len(h),512):out.append(design(m.velocity,torch.tensor(np.c_[h[j:j+512],p[j:j+512]/42000],dtype=torch.float64)).cpu().numpy())
 del b;torch.cuda.empty_cache();return np.concatenate(out)

if __name__=='__main__':
 out=R/'evidence/duration/features';out.mkdir(parents=True,exist_ok=True)
 for ds in ['public','legacy']:
  from native_data import load_native
  z=load_native(ds);valid=z['valid60'];train=z['train'][valid[z['train']]];val=z['val'][valid[z['val']]];ids=np.r_[train,val];X=z['X60'][ids]
  for N in [10,20,60]:
   for method in ['GDL','B-spline-KAN-2025','RGA-KAN-2026','TimeFormer-2026']:
    p=out/f'{ds}_{N}_{method}.npz'
    if p.exists():print('REUSE',p.name,flush=True);continue
    feature=extract(ds,method,X,N);np.savez_compressed(p,indices=ids,n_train=len(train),**feature);print('FEATURES',p.name,len(ids),flush=True)
 print('DURATION FEATURES COMPLETE',flush=True)
