"""Cache training/validation observation features, never construct new observations."""
import torch,numpy as np
from data_v7 import ROOT,BASE,load_data,observer_frame,kinematics
from stable_inference import load_corrected
from original_model_utils import encode
torch.set_num_threads(4)
X,y,_,idx=load_data('legacy');ids=np.r_[idx['train'],idx['val']];b=load_corrected(BASE/'checkpoints/v5/legacy__GDL.pt',device='cpu');norm=b['norm'];enc=b['encoder'];H=[]
with torch.no_grad():
 for j in range(0,len(ids),512):
  x=(X[ids[j:j+512]].reshape(-1,240).astype(float)-norm['X_mean'])/(norm['X_std']+1e-12);H.append(encode(enc,torch.tensor(x,dtype=torch.float64)))
 H=torch.cat(H);rho=enc.f1(H)*enc.rho_std+enc.rho_mean
xx,yy,_=observer_frame(X[ids],y[ids]);f,dc,dl=kinematics(xx);raw=xx.copy();raw[:,:,:3]/=42000
torch.save(dict(H=H,oldrho=rho,f=torch.tensor(f),cam=torch.tensor(xx[:,-1,:3],dtype=torch.float64),los=torch.tensor(xx[:,-1,3:],dtype=torch.float64),dc=torch.tensor(dc),dl=torch.tensor(dl),y=torch.tensor(yy,dtype=torch.float64),raw=torch.tensor(raw.reshape(-1,240),dtype=torch.float64),n_train=len(idx['train']),state_scales=[float(np.linalg.norm(y[idx['train'],:3].astype(float).std(0))),float(np.linalg.norm(y[idx['train'],3:].astype(float).std(0)))]),ROOT/'evidence/observer_features.pt')
