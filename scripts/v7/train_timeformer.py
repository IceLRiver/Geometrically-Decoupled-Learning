from pathlib import Path
import sys,json,time,copy
import numpy as np
import torch
from torch import nn
from data_v7 import ROOT,load_data,observer_frame
from timeformer import TimeFormer
torch.set_num_threads(4)
OUT=ROOT/'evidence/timeformer_formal';OUT.mkdir(exist_ok=True)
for ds in ['public','legacy']:
 X,y,_,idx=load_data(ds);X,y,B=observer_frame(X,y);tr=idx['train'];va=idx['val'];mu=X[tr].mean((0,1));std=np.maximum(X[tr].std((0,1)),1e-5);ym=y[tr].mean(0);ys=np.maximum(y[tr].std(0),1e-3);xx=torch.tensor((X-mu)/std,device='cuda',dtype=torch.float32);yy=torch.tensor((y-ym)/ys,device='cuda',dtype=torch.float32);ys_t=torch.tensor(ys,device='cuda');yval=torch.tensor(y[va],device='cuda',dtype=torch.float32);sc=[np.linalg.norm(y[tr,:3].std(0)),np.linalg.norm(y[tr,3:].std(0))]
 for width,scales in [(16,1),(16,2),(32,2)]:
  stem=f'{ds}_w{width}s{scales}';torch.manual_seed(42);rng=np.random.default_rng(42);model=TimeFormer(width,scales).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');stale=0;hist=[];start=time.time()
  for epoch in range(150):
   model.train();order=rng.permutation(tr)
   for j in range(0,len(order),1024):
    ids=order[j:j+1024];loss=((model(xx[ids])-yy[ids])**2).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
   model.eval();p=[]
   with torch.no_grad():
    for j in range(0,len(va),512):p.append(model(xx[va[j:j+512]]))
    delta=(torch.cat(p)-yy[va])*ys_t;er=delta[:,:3].square().sum(1).mean().sqrt().item();ev=delta[:,3:].square().sum(1).mean().sqrt().item()
   score=er/sc[0]+ev/sc[1];row=dict(epoch=epoch+1,position_km_rmse=er,velocity_km_s_rmse=ev,score=score,seconds=time.time()-start);hist.append(row)
   if score<best:
    best=score;stale=0;torch.save(dict(state=model.state_dict(),config=dict(width=width,scales=scales,gamma=.1,dataset=ds),mu=mu,std=std,ym=ym,ys=ys,validation=row,parameters=sum(p.numel() for p in model.parameters())),OUT/(stem+'.pt'))
   else:stale+=1
   (OUT/(stem+'_history.json')).write_text(json.dumps(hist));print(json.dumps(dict(run=stem,**row)),flush=True)
   if stale>=25:break
 print('DATASET VALIDATION SEARCH COMPLETE',ds,flush=True)
