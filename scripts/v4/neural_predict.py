"""Checkpoint inference accepts observations only; no target labels enter this API."""
import time
import numpy as np
import torch
from train_models import Model
from benchmark_data import kinematics,observer_frame
STAT_KEYS=['f_mean','f_std','p_mean','p_std','v_mean','v_std','res_mean','res_std','rho_mean','rho_std','rho_scale','radius_mean']

def load_checkpoint(path,device='cuda'):
 checkpoint=torch.load(path,map_location='cpu');c=checkpoint['config'];stats={k:checkpoint['state'][k].cpu().numpy() for k in STAT_KEYS if k in checkpoint['state']};model=Model(c['kind'],stats,c.get('width',64),c.get('depth',2),c.get('temporal_stride',1)).to(device);model.load_state_dict(checkpoint['state']);model.eval();return model,c

def predict(model,config,X,batch_size=128):
 X=np.asarray(X);coord=config.get('coordinates','inertial');extra=None;B=None
 if coord=='hybrid':
  _,dv,_=kinematics(X);extra=np.c_[X[:,0,:3]/42000,dv/8]
 if coord in ['observer','hybrid']:X,_,B=observer_frame(X,np.zeros((len(X),6)))
 f,dc,dl=kinematics(X,config.get('degree',2))
 if extra is not None:f=np.c_[f,extra]
 device=next(model.parameters()).device;p=[];v=[]
 with torch.no_grad():
  for j in range(0,len(X),batch_size):
   tensors=[torch.as_tensor(z[j:j+batch_size],device=device,dtype=torch.float32) for z in [X,f,dc,dl]];pp,vv,_=model(*tensors);p.append(pp.cpu().numpy());v.append(vv.cpu().numpy())
 p=np.concatenate(p).astype(float);v=np.concatenate(v).astype(float)
 if B is not None:p=np.einsum('nbi,nb->ni',B,p);v=np.einsum('nbi,nb->ni',B,v)
 return np.c_[p,v]

def metrics(prediction,truth):
 finite=np.isfinite(prediction).all(1);d=prediction[finite]-truth[finite];out={'arcs':len(prediction),'finite':int(finite.sum()),'finite_fraction':float(finite.mean())}
 for key,e in [('position_km',np.linalg.norm(d[:,:3],axis=1)),('velocity_km_s',np.linalg.norm(d[:,3:],axis=1))]:
  for label,value in [('rmse',np.sqrt(np.mean(e**2))),('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95))]:out[key+'_'+label]=float(value)
 return out
