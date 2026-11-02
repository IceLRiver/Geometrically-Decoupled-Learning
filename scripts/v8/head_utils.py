import numpy as np
import torch
RIDGES=[1e-6,1e-4,.01,1.,100.]
def extras(c,cross,stats=None,n=None):
 f=np.asarray(c['f'],float)
 if stats is None:stats=dict(mean=f[:n].mean(0),std=np.maximum(f[:n].std(0),1e-7))
 z=np.clip((f-stats['mean'])/stats['std'],-12,12);a=[f,np.asarray(c['raw'],float)]
 if cross:
  ij=np.triu_indices(f.shape[1]);a.append(z[:,ij[0]]*z[:,ij[1]])
 return np.concatenate(a,1),stats
def fits(phi,Y,n):
 phi=torch.tensor(phi,dtype=torch.float64);Y=torch.tensor(Y,dtype=torch.float64);mu=phi[:n].mean(0);std=phi[:n].std(0).clamp_min(1e-7);z=(phi-mu)/std;ym=Y[:n].mean(0);ys=Y[:n].std(0).clamp_min(1e-3);target=(Y[:n]-ym)/ys;A=z[:n].T@z[:n]/n;b=z[:n].T@target/n;values,vectors=torch.linalg.eigh(A);rhs=vectors.T@b
 for ridge in RIDGES:
  w=vectors@(rhs/(values.clamp_min(0)[:,None]+ridge));p=(z@w)*ys+ym;yield p.numpy(),dict(w=w,mean=mu,std=std,output_mean=ym,output_std=ys,ridge=ridge)
def apply(conf,phi):
 return (((torch.tensor(phi,dtype=torch.float64)-conf['mean'])/conf['std'])@conf['w']*conf['output_std']+conf['output_mean']).numpy()
def errors(p,v,Y,n,sc):
 er=float(np.sqrt(np.mean(np.sum((p[n:]-Y[n:,:3])**2,1))));ev=float(np.sqrt(np.mean(np.sum((v[n:]-Y[n:,3:])**2,1))));return dict(position_km_rmse=er,velocity_km_s_rmse=ev,common_score=er/sc[0]+ev/sc[1])
