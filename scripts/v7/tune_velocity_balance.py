"""Validation-only loss balancing; frozen encoder and unchanged comparisons."""
import json,time,copy,sys
import numpy as np
import torch
from torch import nn
from data_v7 import ROOT
from train_direct_velocity_heads import CorrectedHeads
from rate_heads import RateHeads
torch.set_num_threads(4)
c=torch.load(ROOT/'evidence/observer_features.pt',map_location='cpu');n=c['n_train'];F=torch.cat([c['H'],c['f'],c['raw'],c['oldrho']/42000],1).float().cuda();OR=c['oldrho'].float().cuda();C=c['cam'].float().cuda();L=c['los'].float().cuda();L/=L.norm(dim=1,keepdim=True);DC=c['dc'].float().cuda();DL=c['dl'].float().cuda();Y=c['y'].float().cuda();rho=(Y[:,:3]-C).norm(dim=1,keepdim=True);tr=np.arange(n);va=np.arange(n,len(F));out=ROOT/'evidence/velocity_focused_heads';out.mkdir(exist_ok=True)
strong='--stronger' in sys.argv
for is_rate,weight in ([(True,10000.)] if strong else [(True,100.),(True,1000.),(False,1000.)]):
 stem=('rate' if is_rate else 'direct')+'_v'+str(int(weight));parent=ROOT/'evidence/velocity_focused_heads/rate_v1000.pt' if strong else ROOT/'evidence/observer_full_heads'/('rate_w128.pt' if is_rate else 'w128.pt');ck=torch.load(parent,map_location='cpu');stats={k:ck['state'][k] for k in ['f_mean','f_std','rho_res_std','vres_mean','vres_std']};m=(RateHeads if is_rate else CorrectedHeads)(stats,128).cuda();m.load_state_dict(ck['state']);rng=np.random.default_rng(42);warmup=0 if strong else 20;limit=80 if strong else 100;jointlr=.00005 if strong else .0001
 for p in m.parameters():p.requires_grad_(False)
 for p in m.velocity.parameters():p.requires_grad_(True)
 opt=torch.optim.AdamW(m.velocity.parameters(),lr=.001,weight_decay=.0001);hist=[];best=float('inf');stale=0;start=time.time()
 for ep in range(limit):
  if ep==warmup:
   for p in m.parameters():p.requires_grad_(True)
   opt=torch.optim.AdamW(m.parameters(),lr=jointlr,weight_decay=.0001);stale=0
  m.train();order=rng.permutation(tr)
  for j in range(0,n,1024):
   k=order[j:j+1024];p,v,r=m(F[k],OR[k],C[k],L[k],DC[k],DL[k]);vl=(((v-Y[k,3:])/m.vres_std)**2).mean();loss=vl if ep<warmup else weight*vl+(((r-rho[k])/m.rho_res_std)**2).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],5);opt.step()
  m.eval();er=ev=0.
  with torch.no_grad():
   for j in range(0,len(va),1024):
    k=va[j:j+1024];p,v,_=m(F[k],OR[k],C[k],L[k],DC[k],DL[k]);er+=(p-Y[k,:3]).square().sum().item();ev+=(v-Y[k,3:]).square().sum().item()
  er=(er/len(va))**.5;ev=(ev/len(va))**.5;score=er/c['state_scales'][0]+ev/c['state_scales'][1];row=dict(epoch=ep+1,position_km_rmse=er,velocity_km_s_rmse=ev,common_score=score,phase='velocity warmup' if ep<warmup else 'joint',seconds=time.time()-start);hist.append(row)
  if score<best:
   best=score;stale=0;config=dict(ck['config']);config.update(velocity_weight=weight,warmup_epochs=warmup,epochs=limit,lr_velocity=.001,lr_joint=jointlr,parent_checkpoint=str(parent.relative_to(ROOT)));torch.save(dict(state=copy.deepcopy(m.state_dict()),width=128,validation=row,config=config),out/(stem+'.pt'))
  else:stale+=1
  (out/(stem+'_history.json')).write_text(json.dumps(hist));print(json.dumps(dict(run=stem,**row)),flush=True)
  if ep>=40 and stale>=25:break
print('VELOCITY BALANCE TRAINING COMPLETE',flush=True)
