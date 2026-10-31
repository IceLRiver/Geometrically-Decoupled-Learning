"""FP64 execution of fixed learned parameters to stabilize affine basis sums.
Weights, partitions and validation-selected configurations are unchanged.
"""
from pathlib import Path
import numpy as np
import torch
from data_v5 import BASE,kinematics,observer_frame
from recent_models import BSplineKAN,RGAKAN
from train_recent import features
from neural_predict import load_checkpoint as original_load
from original_model_utils import mod
from original_model_utils import encode
from train_original_correction import CorrectedHeads

def load_checkpoint(path,device='cuda'):
 m,c=original_load(path,device);return m.double().eval(),c

def predict(model,config,X,batch_size=128):
 X=np.asarray(X);coord=config.get('coordinates','inertial');extra=None;B=None
 if coord=='hybrid':_,dv,_=kinematics(X);extra=np.c_[X[:,0,:3]/42000,dv/8]
 if coord in ['observer','hybrid']:X,_,B=observer_frame(X,np.zeros((len(X),6)))
 f,dc,dl=kinematics(X,config.get('degree',2))
 if extra is not None:f=np.c_[f,extra]
 device=next(model.parameters()).device;out=[]
 with torch.no_grad():
  for j in range(0,len(X),batch_size):
   p,v,_=model(*[torch.tensor(a[j:j+batch_size],device=device,dtype=torch.float64) for a in [X,f,dc,dl]]);out.append(np.c_[p.cpu().numpy(),v.cpu().numpy()])
 state=np.concatenate(out)
 if B is not None:state=np.c_[np.einsum('nbi,nb->ni',B,state[:,:3]),np.einsum('nbi,nb->ni',B,state[:,3:])]
 return state

def load_recent(path,device='cuda'):
 c=torch.load(path,map_location='cpu')
 with torch.enable_grad():m=BSplineKAN(270,6,c['width'],5) if c['method']=='B-spline-KAN-2025' else RGAKAN(270,6,c['width'],5,2,c['config'].get('glorot_sine',False))
 m.load_state_dict(c['state']);m=m.to(device).double().eval();return dict(model=m,checkpoint=c,device=device)

def predict_recent(bundle,X,batch_size=512):
 c=bundle['checkpoint'];xx,_,B=observer_frame(X,np.zeros((len(X),6)));f=np.clip((features(xx)-c['mu'])/c['std'],-6,6);out=[]
 with torch.no_grad():
  for j in range(0,len(X),batch_size):out.append(bundle['model'](torch.tensor(f[j:j+batch_size],device=bundle['device'],dtype=torch.float64)).cpu().numpy())
 p=np.concatenate(out)*c['ys']+c['ym'];return np.c_[np.einsum('nbi,nb->ni',B,p[:,:3]),np.einsum('nbi,nb->ni',B,p[:,3:])]

def load_corrected(path,encoder_path=None,norm_path=None,device='cuda'):
 encoder_path=Path(encoder_path) if encoder_path else BASE/'checkpoints/v5/original_64d4.pth';norm_path=Path(norm_path) if norm_path else BASE/'data/legacy_simulation/norm_stats_legacy.npz';s=torch.load(encoder_path,map_location='cpu');keys=['cam_mean','cam_std','los_mean','los_std','rho_mean','rho_std','pos_mean','pos_std','vel_mean','vel_std'];stats={k:s[k] for k in keys};stats['rho_mean']=float(stats['rho_mean']);stats['rho_std']=float(stats['rho_std']);encoder=mod.OrbitPhysKAN(d_model=64,num_mid_layers=4,norm_params=stats);encoder.load_state_dict(s);encoder=encoder.to(device).double().eval();c=torch.load(path,map_location='cpu');stats={k:c['state'][k] for k in ['f_mean','f_std','rho_res_std','vres_mean','vres_std']};head=CorrectedHeads(stats,c['width']);head.load_state_dict(c['state']);head=head.to(device).double().eval();return dict(encoder=encoder,head=head,norm=dict(np.load(norm_path)),device=device)

def predict_corrected(bundle,X,batch_size=512):
 X=np.asarray(X);feature,dc,dl=kinematics(X);norm=bundle['norm'];device=bundle['device'];encoder=bundle['encoder'];out=[]
 with torch.no_grad():
  for j in range(0,len(X),batch_size):
   x=X[j:j+batch_size];xx=(x.reshape(-1,240).astype(float)-norm['X_mean'])/(norm['X_std']+1e-12);h=encode(encoder,torch.tensor(xx,device=device,dtype=torch.float64));oldrho=encoder.f1(h)*encoder.rho_std+encoder.rho_mean;f=torch.cat([h,torch.tensor(feature[j:j+batch_size],device=device,dtype=torch.float64),oldrho/42000],1);cam=torch.tensor(x[:,-1,:3],device=device,dtype=torch.float64);los=torch.tensor(x[:,-1,3:],device=device,dtype=torch.float64);los/=los.norm(dim=1,keepdim=True);p,v,_=bundle['head'](f,oldrho,cam,los,torch.tensor(dc[j:j+batch_size],device=device,dtype=torch.float64),torch.tensor(dl[j:j+batch_size],device=device,dtype=torch.float64));out.append(np.c_[p.cpu().numpy(),v.cpu().numpy()])
 return np.concatenate(out)
