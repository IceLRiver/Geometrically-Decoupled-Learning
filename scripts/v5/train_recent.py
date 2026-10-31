from pathlib import Path
import os,sys,json,time,copy,hashlib,argparse
import numpy as np
import torch
from torch import nn
from data_v5 import ROOT,BASE,load_data,kinematics,observer_frame,protocol
from recent_models import BSplineKAN,RGAKAN
torch.set_num_threads(4);DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def features(X):
 f,_,_=kinematics(X);raw=np.array(X,dtype=np.float64,copy=True);raw[:,:,:3]/=42000
 return np.c_[raw.reshape(len(X),-1),f].astype('float32')

def predict(model,F,ym,ys,ids,B):
 model.eval();out=[]
 with torch.no_grad():
  for j in range(0,len(ids),512):out.append(model(F[ids[j:j+512]]).cpu().numpy())
 state=np.concatenate(out).astype(float)*ys+ym
 return np.c_[np.einsum('nbi,nb->ni',B[ids],state[:,:3]),np.einsum('nbi,nb->ni',B[ids],state[:,3:])]

def train(ds,method,width,epochs=150,glorot_sine=False):
 torch.manual_seed(42);rng=np.random.default_rng(42);X,y,_,idx=load_data(ds);truth=np.asarray(y,dtype=float);X,y,B=observer_frame(X,y);f=features(X);tr=idx['train'];va=idx['val'];mu=f[tr].mean(0);std=np.maximum(f[tr].std(0),1e-4);ym=y[tr].mean(0);ys=np.maximum(y[tr].std(0),1e-3);F=torch.as_tensor(np.clip((f-mu)/std,-6,6),device=DEVICE);Y=torch.as_tensor((y-ym)/ys,device=DEVICE,dtype=torch.float32);model=(BSplineKAN(f.shape[1],6,width,5) if method=='B-spline-KAN-2025' else RGAKAN(f.shape[1],6,width,5,2,glorot_sine)).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scales=[np.linalg.norm(truth[tr,:3].std(0)),np.linalg.norm(truth[tr,3:].std(0))];out=ROOT/'evidence/recent_models';out.mkdir(parents=True,exist_ok=True);stem=f'{ds}__{method}_w{width}_s42'+('_glorot' if glorot_sine else '');history=[];best=np.inf;stale=0;start=time.time()
 for epoch in range(epochs):
  model.train();order=rng.permutation(tr)
  for j in range(0,len(order),1024):
   ids=order[j:j+1024];loss=((model(F[ids])-Y[ids])**2).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  p=predict(model,F,ym,ys,va,B);er=np.sqrt(np.mean(np.sum((p[:,:3]-truth[va,:3])**2,axis=1)));ev=np.sqrt(np.mean(np.sum((p[:,3:]-truth[va,3:])**2,axis=1)));score=er/scales[0]+ev/scales[1];row=dict(epoch=epoch+1,validation_position_km_rmse=float(er),validation_velocity_km_s_rmse=float(ev),selection_score=float(score),seconds=time.time()-start);history.append(row)
  if score<best:
   best=score;stale=0;torch.save(dict(state=model.state_dict(),method=method,width=width,D=5,blocks=2,mu=mu,std=std,ym=ym,ys=ys,config=dict(dataset=ds,seed=42,epochs=epochs,lr=.001,batch_size=1024,weight_decay=.0001,coordinates='observer',clip_input=6,glorot_sine=glorot_sine),best_epoch=epoch+1,validation=row,source='ICLR2025 KAN / CMAME2026 RGA-KAN regression adaptation'),out/(stem+'.pt'))
  else:stale+=1
  (out/(stem+'_history.json')).write_text(json.dumps(history));print(json.dumps(dict(run=stem,**row)),flush=True)
  if stale>=25:break
 return dict(checkpoint=(out/(stem+'.pt')).relative_to(ROOT).as_posix(),score=float(best))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=['public','legacy','simulation']);ap.add_argument('--methods',nargs='+',default=['B-spline-KAN-2025','RGA-KAN-2026']);ap.add_argument('--widths',nargs='+',type=int,default=[32,64]);ap.add_argument('--epochs',type=int,default=150);ap.add_argument('--glorot-sine',action='store_true');a=ap.parse_args();protocol();out=ROOT/'evidence/recent_models';out.mkdir(parents=True,exist_ok=True)
 sel_suffix='_glorot' if a.glorot_sine else ''
 for ds in a.datasets:
  for method in a.methods:
   records=[]
   for width in a.widths:records.append(train(ds,method,width,a.epochs,a.glorot_sine))
   selected=dict(min(records,key=lambda x:x['score']));selected['sha256']=hashlib.sha256((ROOT/selected['checkpoint']).read_bytes()).hexdigest();selected['candidates']=records;(out/f'{ds}__{method}{sel_suffix}_selected.json').write_text(json.dumps(selected,indent=2));print('FROZEN',ds,method,selected,flush=True)
