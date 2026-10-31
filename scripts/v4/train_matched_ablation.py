import os
"""Common-capacity component study: identical inputs, widths, optimizer and budget."""
import json,time,copy
import numpy as np
import torch
from torch import nn
from train_models import Model,tensors,metrics,DEVICE
from benchmark_data import ROOT,load_data,observer_frame

def run(ds,kind,lam):
 X,y,_,idx=load_data(ds);X,y,_=observer_frame(X,y);T,stats=tensors(X,y,idx,2,kind);torch.manual_seed(42);rng=np.random.default_rng(42);model=Model(kind,stats,32,1,8).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);scale=model.res_std;stem=f'mini_{ds}_{kind}_s42';out=ROOT/'evidence/matched_ablation';out.mkdir(exist_ok=True);best=np.inf;stale=0;history=[];begin=time.time();cfg=dict(dataset=ds,kind=kind,width=32,depth=1,degree=2,coordinates='observer',temporal_stride=8,batch_size=1024,epochs=100,lr=.001,weight_decay=.0001,range_weight=lam,velocity_scale='residual',seed=42)
 for epoch in range(100):
  model.train();perm=rng.permutation(idx['train'])
  for j in range(0,len(perm),1024):
   k=perm[j:j+1024];p,v,rho=model(*[a[k] for a in T[:4]]);loss=(((v-T[4][k,3:])/scale)**2).mean();loss+=(((p-T[4][k,:3])/model.p_std)**2).mean() if kind.startswith('Direct') else lam*(((rho-T[5][k])/model.rho_std)**2).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  val,_,_=metrics(model,T,idx['val']);score=val['position_km_rmse']/float(stats['rho_std'][0])+val['velocity_km_s_rmse']/float(np.linalg.norm(scale.cpu().numpy()));row=dict(epoch=epoch+1,validation=val,selection_score=score);history.append(row)
  if score<best:
   best=score;stale=0;state=copy.deepcopy(model.state_dict());torch.save(dict(state=state,config=cfg,validation=val,selection_score=score,best_epoch=epoch+1),out/(stem+'.pt'))
  else:stale+=1
  (out/(stem+'_history.json')).write_text(json.dumps(history));print(json.dumps(dict(run=stem,**row,elapsed_s=time.time()-begin)),flush=True)
  if stale>=20:break
if __name__=='__main__':
 for ds in (['public','simulation']+(['real'] if os.environ.get('ORBIT_REAL_DATA') else [])):
  for kind,lam in [('Geometric-KAN',1.),('Direct-KAN',1.),('Geometric-MLP',1.),('NoRange-KAN',0.),('FixedFrequency-KAN',1.),('Original-KAN',1.)]:run(ds,kind,lam)
