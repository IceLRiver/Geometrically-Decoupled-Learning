"""Common training-only head adaptation at non-anchor observation lengths."""
from pathlib import Path
import os,json,time
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import torch
from duration_features import R,B,velocity_design
torch.set_num_threads(4)
E=R/'evidence/duration';OUT=E/'adapted';OUT.mkdir(parents=True,exist_ok=True)
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
for ds in ['public','legacy']:
 from native_data import load_native
 raw=load_native(ds);truth=raw['y'];tr=raw['train'];sc=[np.linalg.norm(truth[tr,:3].astype(float).std(0)),np.linalg.norm(truth[tr,3:].astype(float).std(0))]
 for N in [10,20,60]:
  for method in ['GDL','B-spline-KAN-2025','RGA-KAN-2026','TimeFormer-2026']:
   target=OUT/f'{ds}_{N}_{method}.pt'
   if target.exists():print('REUSE',target.name,flush=True);continue
   path=E/'features'/f'{ds}_{N}_{method}.npz'
   while not path.exists():time.sleep(10)
   c=np.load(path);n=int(c['n_train']);ids=c['indices'];Y=np.c_[np.einsum('nbi,ni->nb',c['B'],truth[ids,:3]),np.einsum('nbi,ni->nb',c['B'],truth[ids,3:])];best=None;records=[]
   for cross in [False,True]:
    ex,exstats=extras(c,cross,n=n);phi=np.c_[c['phi'],ex]
    if method=='GDL':
     rho=np.linalg.norm(Y[:,:3]-c['cam'],axis=1,keepdims=True);bestR=None
     for r,rc in fits(phi,rho,n):
      r=np.logaddexp(0,r);p=c['cam']+r*c['los'];er=np.sqrt(np.mean(np.sum((p[n:]-Y[n:,:3])**2,1)))
      if bestR is None or er<bestR[0]:bestR=(er,r,rc,p)
     _,r,rc,p=bestR;pv=velocity_design(ds,c['h'],p);vphi=np.c_[pv,ex];anchor=c['dc']+r*c['dl'];vy=((Y[:,3:]-anchor)*c['los']).sum(1,keepdims=True) if ds=='legacy' else Y[:,3:]-anchor
     for residual,vc in fits(vphi,vy,n):
      v=anchor+residual*c['los'] if ds=='legacy' else anchor+residual;val=errors(p,v,Y,n,sc);records.append(dict(cross=cross,range_ridge=rc['ridge'],velocity_ridge=vc['ridge'],**val))
      if best is None or val['common_score']<best['validation']['common_score']:best=dict(dataset=ds,method=method,measurements=N,cross=cross,extra_stats=exstats,range=rc,velocity=vc,range_rate=ds=='legacy',validation=val,training_rows=n,validation_rows=len(ids)-n)
    else:
     for state,head in fits(phi,Y,n):
      val=errors(state[:,:3],state[:,3:],Y,n,sc);records.append(dict(cross=cross,ridge=head['ridge'],**val))
      if best is None or val['common_score']<best['validation']['common_score']:best=dict(dataset=ds,method=method,measurements=N,cross=cross,extra_stats=exstats,head=head,validation=val,training_rows=n,validation_rows=len(ids)-n)
   torch.save(best,target);(OUT/(target.stem+'.json')).write_text(json.dumps(dict(validation=best['validation'],cross=best['cross'],training_rows=n,candidates=records),indent=2));print(target.stem,best['validation'],flush=True)
 print('SOURCE HEAD ADAPTATION COMPLETE',ds,flush=True)
print('ALL DURATION HEADS COMPLETE',flush=True)
