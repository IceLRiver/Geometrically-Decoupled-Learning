from pathlib import Path
import os,json
import numpy as np
BASE=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('ORBIT_RUN_ROOT',str(BASE/'runs/v4'))).resolve()
(ROOT/'evidence').mkdir(parents=True,exist_ok=True)
def load_data(dataset):
 if dataset=='simulation':
  folder=BASE/'data/multiregime';manifest=json.loads((folder/'manifest.json').read_text());xs=[];ys=[];meta=[]
  for item in manifest['chunks']:
   with np.load(folder/item['file']) as z:
    xs.append(z['X']);ys.append(z['y']);meta.extend({'target':int(k),'split':['train','val','test'][int(c)]} for k,c in zip(z['orbit_id'],z['split']))
  X,y=np.concatenate(xs),np.concatenate(ys)
 elif dataset=='public':
  folder=BASE/'data/sentinel1_poeorb'
  with np.load(folder/'observations.npz') as z:X,y=z['X'],z['y']
  meta=json.loads((folder/'arc_metadata.json').read_text())
 elif dataset=='real':
  location=os.environ.get('ORBIT_REAL_DATA')
  if not location:raise RuntimeError('Set ORBIT_REAL_DATA to your private directory containing real_arcs.npz, real_arcs_meta.json and real_split.json.')
  folder=Path(location).expanduser().resolve()
  with np.load(folder/'real_arcs.npz') as z:X,y=z['X'],z['y']
  meta=json.loads((folder/'real_arcs_meta.json').read_text());split=json.loads((folder/'real_split.json').read_text())
  for s,key in [('train','train_indices'),('val','validation_indices'),('test','test_indices')]:
   for i in split[key]:meta[i]['split']=s
 else:raise ValueError(dataset)
 idx={s:np.flatnonzero([m['split']==s for m in meta]) for s in ['train','val','test']};groups=[set(str(meta[i]['target']) for i in idx[s]) for s in ['train','val','test']]
 assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
 return X,y,meta,idx

def kinematics(X,degree=2):
 tau=(np.arange(40)-39)/39.;A=np.stack([tau**i for i in range(4)],1);P=np.linalg.pinv(A)
 c=np.einsum('ij,njk->nik',P,X[:,:,:3].astype(float));dc=c[:,1]/39;ddc=2*c[:,2]/39**2
 coeff=np.einsum('ij,njk->nik',np.linalg.pinv(A[:,:degree+1]),X[:,:,3:].astype(float));dl=coeff[:,1]/39;los=X[:,-1,3:];dl-=np.sum(dl*los,axis=1,keepdims=True)*los
 dd=2*coeff[:,2]/39**2 if degree>1 else np.zeros_like(dl)
 f=np.concatenate([X[:,[0,19,39],:3].reshape(-1,9)/42000,X[:,[0,19,39],3:].reshape(-1,9),dc/8,ddc/.01,dl/.001,dd/.00001],1)
 return f.astype('float32'),dc.astype('float32'),dl.astype('float32')

def observer_frame(X,y):
 """Constant per-arc rotation derived entirely from observed observer positions."""
 r=X[:,-1,:3].astype(float);u=r/np.linalg.norm(r,axis=1,keepdims=True);_,v,_=kinematics(X);t=v-np.sum(v*u,axis=1,keepdims=True)*u;t/=np.linalg.norm(t,axis=1,keepdims=True);n=np.cross(u,t);B=np.stack([u,t,n],axis=1)
 xx=np.concatenate([np.einsum('nbi,nti->ntb',B,X[:,:,:3]),np.einsum('nbi,nti->ntb',B,X[:,:,3:])],axis=-1)
 yy=np.concatenate([np.einsum('nbi,ni->nb',B,y[:,:3]),np.einsum('nbi,ni->nb',B,y[:,3:])],axis=-1)
 return xx,yy,B

def freeze_protocol():
 path=BASE/'data/benchmark_protocol_v4.json';p=json.loads(path.read_text())
 if os.environ.get('ORBIT_REAL_DATA'):
  X,y,m,idx=load_data('real');rng=np.random.default_rng(p['seed']);p['datasets']['real']={'arcs':len(X),'train':len(idx['train']),'validation':len(idx['val']),'test_full':len(idx['test']),'test_primary':idx['test'].tolist(),'validation_solver':sorted(rng.choice(idx['val'],min(30,len(idx['val'])),replace=False).tolist())}
 return p
