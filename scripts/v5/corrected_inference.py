"""Observation-only inference for the original-encoder geometric refinement."""
from pathlib import Path
import numpy as np
import torch
from data_v5 import BASE,kinematics
from original_model_utils import mod
from original_model_utils import encode
from train_original_correction import CorrectedHeads

def load_corrected(path,encoder_path=None,norm_path=None,device=None):
 device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
 encoder_path=Path(encoder_path) if encoder_path else BASE/'checkpoints/v5/original_64d4.pth'
 norm_path=Path(norm_path) if norm_path else BASE/'data/legacy_simulation/norm_stats_legacy.npz'
 s=torch.load(encoder_path,map_location='cpu');keys=['cam_mean','cam_std','los_mean','los_std','rho_mean','rho_std','pos_mean','pos_std','vel_mean','vel_std'];params={k:s[k] for k in keys};params['rho_mean']=float(params['rho_mean']);params['rho_std']=float(params['rho_std']);encoder=mod.OrbitPhysKAN(d_model=64,num_mid_layers=4,norm_params=params).to(device).eval();encoder.load_state_dict(s);c=torch.load(path,map_location='cpu');stats={k:c['state'][k] for k in ['f_mean','f_std','rho_res_std','vres_mean','vres_std']};head=CorrectedHeads(stats,c['width']).to(device).eval();head.load_state_dict(c['state']);norm=dict(np.load(norm_path));return dict(encoder=encoder,head=head,norm=norm,checkpoint=c)

def inputs(bundle,X):
 device=next(bundle['head'].parameters()).device
 X=np.asarray(X);feature,dc,dl=kinematics(X);H=[];norm=bundle['norm'];encoder=bundle['encoder']
 with torch.no_grad():
  for j in range(0,len(X),512):
   xx=(X[j:j+512].reshape(-1,240).astype(float)-norm['X_mean'])/(norm['X_std']+1e-12);H.append(encode(encoder,torch.tensor(xx,device=device,dtype=torch.float32)))
  H=torch.cat(H);oldrho=encoder.f1(H)*encoder.rho_std+encoder.rho_mean;f=torch.cat([H,torch.tensor(feature,device=device),oldrho/42000],1)
  cam=torch.tensor(X[:,-1,:3],device=device,dtype=torch.float32);los=torch.tensor(X[:,-1,3:],device=device,dtype=torch.float32);los=los/los.norm(dim=1,keepdim=True)
 return [f,oldrho,cam,los,torch.tensor(dc,device=device),torch.tensor(dl,device=device)]

def predict_corrected(bundle,X,batch_size=512):
 T=inputs(bundle,X);out=[]
 with torch.no_grad():
  for j in range(0,len(X),batch_size):
   p,v,_=bundle['head'](*[a[j:j+batch_size] for a in T]);out.append(np.c_[p.cpu().numpy(),v.cpu().numpy()])
 return np.concatenate(out).astype(float)
