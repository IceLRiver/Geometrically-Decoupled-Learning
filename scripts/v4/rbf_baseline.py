"""Gaussian RBF regression, following the 2024 RBFNN model family.

Inputs are the measured LOS series and observer's initial position/estimated
velocity. Training-only standardization; centres from training-only k-means++.
Output is terminal inertial state (shared evaluation target across methods).
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,time
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from benchmark_data import ROOT,load_data,freeze_protocol,kinematics

def representation_data(X,y,representation='inertial'):
 X=np.asarray(X,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 if representation=='inertial':
  _,dc,_=kinematics(X);return np.c_[X[:,0,:3],dc,X[:,:,3:].reshape(-1,120)],y,None
 tau=(np.arange(40)-39)/39.;A=np.stack([tau**i for i in range(4)],1);coef=np.einsum('ij,njk->nik',np.linalg.pinv(A),X[:,:,:3].astype(float));v=(coef[:,1,None]+2*coef[:,2,None]*tau[None,:,None]+3*coef[:,3,None]*tau[None,:,None]**2)/39.;acc=(2*coef[:,2,None]+6*coef[:,3,None]*tau[None,:,None])/39**2
 R=X[:,:,:3];r=np.linalg.norm(R,axis=-1,keepdims=True);er=R/r;h=np.cross(R,v);hn=np.linalg.norm(h,axis=-1,keepdims=True);en=h/hn;et=np.cross(en,er);B=np.stack([er,et,en],axis=-2)
 erd=(v-np.sum(v*er,axis=-1,keepdims=True)*er)/r;hd=np.cross(R,acc);end=(hd-np.sum(hd*en,axis=-1,keepdims=True)*en)/hn;etd=np.cross(end,er)+np.cross(en,erd);Bd=np.stack([erd,etd,end],axis=-2)
 LOS=np.einsum('ntbi,nti->ntb',B,X[:,:,3:]);angles=np.stack([np.unwrap(np.arctan2(LOS[:,:,1],LOS[:,:,0]),axis=1),np.arcsin(np.clip(LOS[:,:,2],-1,1))],-1);f=np.c_[R[:,0],v[:,0],angles.reshape(len(X),-1)]
 Bend=B[:,-1];Bdot=Bd[:,-1];rel=y[:,:3]-R[:,-1];p=np.einsum('nbi,ni->nb',Bend,rel);vel=np.einsum('nbi,ni->nb',Bend,y[:,3:]-v[:,-1])+np.einsum('nbi,ni->nb',Bdot,rel)
 return f,np.c_[p,vel],(Bend,Bdot,R[:,-1],v[:,-1])

def inverse_state(pred,transform,ids):
 if transform is None:return pred
 B,Bd,R,v=[a[ids] for a in transform];rel=np.einsum('nbi,nb->ni',B,pred[:,:3]);velocity=np.einsum('nbi,nb->ni',B,pred[:,3:]-np.einsum('nbi,ni->nb',Bd,rel))+v;return np.c_[R+rel,velocity]

def run(ds,representation='inertial',normalization='standard'):
 X,y,_,idx=load_data(ds);_,dc,_=kinematics(X)
 f,labels,transform=representation_data(X,y,representation);tr=idx['train'];va=idx['val']
 if normalization=='minmax':mu=(f[tr].max(0)+f[tr].min(0))/2;std=np.maximum((f[tr].max(0)-f[tr].min(0))/2,1e-5)
 else:mu=f[tr].mean(0);std=np.maximum(f[tr].std(0),1e-5)
 f=(f-mu)/std
 ym=labels[tr].mean(0);ys=np.maximum(labels[tr].std(0),1e-3);target=(labels-ym)/ys;eval_scales=[np.linalg.norm(y[tr,:3].std(0)),np.linalg.norm(y[tr,3:].std(0))];rng=np.random.default_rng(42);pool=rng.choice(tr,min(10000,len(tr)),replace=False);best=None;records=[]
 for centers in [64,128,256,512]:
  if centers>=len(pool):continue
  C,_=kmeans2(f[pool],centers,iter=15,minit='++',seed=42,check_finite=True)
  dist=cdist(C,C,'sqeuclidean');base=np.median(dist[dist>0])
  for factor in [.25,1.,4.]:
   width=base*factor;A=np.c_[np.exp(-cdist(f[tr],C,'sqeuclidean')/width),np.ones(len(tr))];Av=np.c_[np.exp(-cdist(f[va],C,'sqeuclidean')/width),np.ones(len(va))];AtA=A.T@A;AtY=A.T@target[tr]
   for ridge in [1e-6,1e-3,.1]:
    W=np.linalg.solve(AtA+ridge*np.eye(centers+1),AtY);p=inverse_state(Av@W*ys+ym,transform,va);er=np.sqrt(np.mean(np.sum((p[:,:3]-y[va,:3])**2,axis=1)));ev=np.sqrt(np.mean(np.sum((p[:,3:]-y[va,3:])**2,axis=1)));score=er/eval_scales[0]+ev/eval_scales[1];rec=dict(centers=centers,width_factor=factor,ridge=ridge,position_km_rmse=er,velocity_km_s_rmse=ev,score=score,representation=representation,normalization=normalization);records.append(rec)
    if best is None or score<best[0]:best=(score,dict(C=C,W=W,width=width,mu=mu,std=std,ym=ym,ys=ys,representation=np.array(representation)),rec)
 folder=ROOT/'evidence/rbf_v4';folder.mkdir(exist_ok=True);stem=ds+('_lvlh' if representation=='lvlh' else '')+('_minmax' if normalization=='minmax' else '');np.savez_compressed(folder/f'{stem}_model.npz',**best[1]);(folder/f'{stem}_validation.json').write_text(json.dumps({'best':best[2],'search':records},indent=2));print(ds,best[2],flush=True)
if __name__=='__main__':
 import sys
 representation='lvlh' if '--lvlh' in sys.argv else 'inertial'
 normalization='minmax' if '--minmax' in sys.argv else 'standard'
 for ds in (['public','simulation']+(['real'] if os.environ.get('ORBIT_REAL_DATA') else [])):run(ds,representation,normalization)
