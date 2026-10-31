"""Observation-only head conditioning, initialized from the supplied original model.
The frozen original encoder is reused; the differentiable scalar-range/position/
velocity hierarchy and orbit-physics KAN edge functions are retained.
"""
from pathlib import Path
import json,time,copy
import numpy as np
import torch
from torch import nn
from data_v5 import ROOT,BASE,load_data,kinematics,protocol
from original_model_utils import mod
from original_model_utils import encode
from fast_kan import FastPhysKANLayer
torch.set_num_threads(4)

class CorrectedHeads(nn.Module):
 def __init__(self,stats,width=64):
  super().__init__()
  for k,v in stats.items():self.register_buffer(k,torch.as_tensor(v,dtype=torch.float32))
  self.embed=FastPhysKANLayer(len(stats['f_mean']),width,32);self.range=FastPhysKANLayer(width,1,32);self.velocity=FastPhysKANLayer(width+3,3,32)
 def forward(self,f,oldrho,cam,los,dc,dl):
  h=self.embed(((f-self.f_mean)/self.f_std).clamp(-12,12));rho=oldrho+self.rho_res_std*self.range(h);p=cam+rho*los;v=dc+rho*dl+self.vres_mean+self.vres_std*self.velocity(torch.cat([h,p/42000],1));return p,v,rho

def main(resume=False,epochs=200,widths=(64,128)):
 start=time.time();X,y,_,idx=load_data('legacy');raw=np.asarray(X,float);features,dc,dl=kinematics(raw);norm=np.load(BASE/'data/legacy_simulation/norm_stats_legacy.npz');s=torch.load(BASE/'checkpoints/v5/original_64d4.pth',map_location='cpu');keys=['cam_mean','cam_std','los_mean','los_std','rho_mean','rho_std','pos_mean','pos_std','vel_mean','vel_std'];params={k:s[k] for k in keys};params['rho_mean']=float(params['rho_mean']);params['rho_std']=float(params['rho_std']);encoder=mod.OrbitPhysKAN(d_model=64,num_mid_layers=4,norm_params=params).cuda().eval();encoder.load_state_dict(s)
 # Test rows are excluded from feature caching during fitting.
 ids=np.r_[idx['train'],idx['val']];tr=np.arange(len(idx['train']));va=np.arange(len(tr),len(ids));T=torch.tensor((X[ids].reshape(-1,240).astype(float)-norm['X_mean'])/(norm['X_std']+1e-12),device='cuda',dtype=torch.float32);H=[]
 with torch.no_grad():
  for j in range(0,len(ids),512):H.append(encode(encoder,T[j:j+512]))
  H=torch.cat(H);oldrho=(encoder.f1(H)*encoder.rho_std+encoder.rho_mean).cpu().numpy()
 f=np.c_[H.cpu().numpy(),features[ids],oldrho/42000];cam=X[ids,-1,:3];los=X[ids,-1,3:];los=los/np.linalg.norm(los,axis=1,keepdims=True);truth=y[ids];rho=np.linalg.norm(truth[:,:3]-cam,axis=1,keepdims=True);res=truth[:,3:]-(dc[ids]+rho*dl[ids]);stats=dict(f_mean=f[tr].mean(0),f_std=np.maximum(f[tr].std(0),1e-4),rho_res_std=np.maximum((rho[tr]-oldrho[tr]).std(0),1e-3),vres_mean=res[tr].mean(0),vres_std=np.maximum(res[tr].std(0),1e-3));F=torch.tensor(f,device='cuda');OR=torch.tensor(oldrho,device='cuda');C=torch.tensor(cam,device='cuda');L=torch.tensor(los,device='cuda');DC=torch.tensor(dc[ids],device='cuda');DL=torch.tensor(dl[ids],device='cuda');Y=torch.tensor(truth,device='cuda');R=torch.tensor(rho,device='cuda');del encoder,T,H
 out=ROOT/'evidence/corrected_original';out.mkdir(exist_ok=True)
 for width in widths:
  torch.manual_seed(42);rng=np.random.default_rng(42);model=CorrectedHeads(stats,width).cuda()
  if resume:model.load_state_dict(torch.load(out/f'w{width}_s42.pt',map_location='cuda')['state'])
  stem=f'w{width}_s42'+('_recovered' if resume else '');opt=torch.optim.AdamW(model.parameters(),lr=.0003 if resume else .001,weight_decay=.0001);best=float('inf');stale=0;history=[]
  for epoch in range(epochs):
   model.train();order=rng.permutation(tr)
   for j in range(0,len(order),1024):
    k=order[j:j+1024];p,v,r=model(F[k],OR[k],C[k],L[k],DC[k],DL[k]);loss=(((r-R[k])/model.rho_res_std)**2).mean()+(((v-Y[k,3:])/model.vres_std)**2).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
   model.eval();er=ev=0
   with torch.no_grad():
    for j in range(0,len(va),1024):
     k=va[j:j+1024];p,v,r=model(F[k],OR[k],C[k],L[k],DC[k],DL[k]);er+=((p-Y[k,:3])**2).sum().item();ev+=((v-Y[k,3:])**2).sum().item()
   er=(er/len(va))**.5;ev=(ev/len(va))**.5;score=er/float(stats['rho_res_std'][0])+ev/float(np.linalg.norm(stats['vres_std']));row=dict(epoch=epoch+1,position_km_rmse=er,velocity_km_s_rmse=ev,selection_score=score,seconds=time.time()-start);history.append(row)
   if score<best:
    best=score;stale=0;torch.save(dict(state=copy.deepcopy(model.state_dict()),width=width,validation=row,config=dict(source_checkpoint='best_model_4layers.pth',encoder_width=64,encoder_depth=4,encoder_frozen=True,seed=42,range_conditioning='original encoder + observation-derived kinematics',epochs=epochs,batch_size=1024,resume_from=f'w{width}_s42.pt' if resume else None,lr=.0003 if resume else .001)),out/(stem+'.pt'))
   else:stale+=1
   (out/(stem+'_history.json')).write_text(json.dumps(history));print(json.dumps(dict(width=width,**row)),flush=True)
   if stale>=30:break
 print('FINISHED: no new test errors computed.',flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--epochs',type=int,default=200);ap.add_argument('--widths',nargs='+',type=int,default=[64,128]);a=ap.parse_args();main(a.resume,a.epochs,a.widths)
