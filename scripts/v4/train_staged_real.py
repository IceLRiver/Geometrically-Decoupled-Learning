"""Validation-selected range pretraining, velocity warm-up, then joint refinement.
The architecture and final differentiable objective are unchanged.
"""
import json,time,copy,argparse
import numpy as np
import torch
from torch import nn
from train_models import Model,tensors,metrics,DEVICE
from benchmark_data import ROOT,load_data,kinematics,observer_frame

def run(coord,width,lam):
 torch.manual_seed(42);rng=np.random.default_rng(42);X,y,_,idx=load_data('real');extra=None
 if coord=='hybrid':
  _,dv,_=kinematics(X);extra=np.c_[X[:,0,:3]/42000,dv/8]
 if coord in ['hybrid','observer']:X,y,_=observer_frame(X,y)
 T,stats=tensors(X,y,idx,2,'Geometric-KAN',extra);model=Model('Geometric-KAN',stats,width,1 if width==32 else 2).to(DEVICE);scale=model.res_std;tr=idx['train'];va=idx['val'];start=time.time();stages=[];history=[];stem=f'real_Geometric-KAN_staged_w{width}_{coord}_lambda{lam}_s42';out=ROOT/'evidence/neural_v4';best=np.inf;stale=0
 def train_epoch(opt,phase):
  model.train();perm=rng.permutation(tr)
  for j in range(0,len(perm),128):
   k=perm[j:j+128];p,v,rho=model(*[z[k] for z in T[:4]]);lr=(((rho-T[5][k])/model.rho_std)**2).mean();lv=(((v-T[4][k,3:])/scale)**2).mean();loss=lr if phase=='range' else lv if phase=='velocity' else lam*lr+lv
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5);opt.step()
 for phase,epochs,rate,patience in [('range',400,.003,60),('velocity',100,.003,30),('joint',150,.0001,35)]:
  for name,param in model.named_parameters():param.requires_grad_(phase!='velocity' or name.startswith('velocity.'))
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=rate,weight_decay=.0001);best=np.inf;stale=0
  for epoch in range(epochs):
   train_epoch(opt,phase);val,_,_=metrics(model,T,va);score=val['position_km_rmse'] if phase=='range' else val['velocity_km_s_rmse'] if phase=='velocity' else val['position_km_rmse']/float(stats['rho_std'][0])+val['velocity_km_s_rmse']/float(np.linalg.norm(scale.cpu().numpy()))
   row={'phase':phase,'epoch':epoch+1,'validation':val,'selection_score':score};stages.append(row)
   if score<best:best=score;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=epoch+1
   else:stale+=1
   if phase=='joint':
    history.append(row)
    if stale==0:
     cfg={'dataset':'real','kind':'Geometric-KAN','width':width,'depth':1 if width==32 else 2,'degree':2,'coordinates':coord,'batch_size':128,'range_weight':lam,'lr':rate,'weight_decay':.0001,'velocity_scale':'residual','seed':42,'training_schedule':'range-pretrain/velocity-warmup/joint'}
     torch.save({'state':state,'config':cfg,'validation':val,'selection_score':score,'best_epoch':best_epoch},out/(stem+'.pt'))
   if epoch%20==0:print(json.dumps({'run':stem,**row,'elapsed_s':time.time()-start}),flush=True)
   (out/(stem+'_stages.json')).write_text(json.dumps(stages));
   if phase=='joint':(out/(stem+'_history.json')).write_text(json.dumps(history))
   if stale>=patience:break
  model.load_state_dict(state)
  print('PHASE DONE',stem,phase,best,flush=True)

if __name__=='__main__':
 for coord,width,lam in [('observer',32,10.),('hybrid',64,10.)]:run(coord,width,lam)
