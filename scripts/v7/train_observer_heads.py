import json,time,copy,sys
import numpy as np
import torch
from torch import nn
from data_v7 import ROOT
from train_direct_velocity_heads import CorrectedHeads
from rate_heads import RateHeads
torch.set_num_threads(4)
def main():
 c=torch.load(ROOT/'evidence/observer_features.pt',map_location='cpu');n=c['n_train'];full='--full-observations' in sys.argv;parts=[c['H'],c['f']]+([c['raw']] if full else [])+[c['oldrho']/42000];F=torch.cat(parts,1).float().cuda();OR=c['oldrho'].float().cuda();C=c['cam'].float().cuda();L=c['los'].float().cuda();L=L/L.norm(dim=1,keepdim=True);DC=c['dc'].float().cuda();DL=c['dl'].float().cuda();Y=c['y'].float().cuda();rho=(Y[:,:3]-C).norm(dim=1,keepdim=True);tr=np.arange(n);va=np.arange(n,len(F));stats=dict(f_mean=F[:n].mean(0),f_std=F[:n].std(0).clamp_min(1e-4),rho_res_std=(rho[:n]-OR[:n]).std(0).clamp_min(1e-3),vres_mean=Y[:n,3:].mean(0),vres_std=Y[:n,3:].std(0).clamp_min(1e-3));out=ROOT/'evidence'/('observer_full_heads' if full else 'observer_heads');out.mkdir(exist_ok=True)
 for width,is_rate in ([(128,False),(128,True)] if full else [(64,False),(128,False),(64,True),(128,True)]):
  runstats=dict(stats);stem=('rate_' if is_rate else '')+f'w{width}'
  if is_rate:
   eta=((Y[:n,3:]-DC[:n]-rho[:n]*DL[:n])*L[:n]).sum(1,keepdim=True);runstats['vres_mean']=eta.mean(0);runstats['vres_std']=eta.std(0).clamp_min(1e-3)
  torch.manual_seed(42);rng=np.random.default_rng(42);m=(RateHeads if is_rate else CorrectedHeads)(runstats,width).float().cuda()
  if full:
   parent=torch.load(ROOT/'evidence/observer_heads'/(stem+'.pt'),map_location='cuda');old=parent['state']
   with torch.no_grad():
    for key,value in m.state_dict().items():
     source=old[key]
     if value.shape==source.shape:value.copy_(source)
     elif value.ndim==2 and source.shape[-1]==95:
      value.zero_();value[:,:94].copy_(source[:,:94]);value[:,-1].copy_(source[:,-1])
     elif value.ndim==1 and len(source)==95:
      if key=='embed.res_gate':value.zero_()
      value[:94].copy_(source[:94]);value[-1].copy_(source[-1])
   print('WARM START',stem,'zero-initialized raw-observation skip channels',flush=True)
  lr=.0003 if full else .001;opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.0001);hist=[];best=float('inf');stale=0;start=time.time()
  for ep in range(250):
   m.train();order=rng.permutation(tr)
   for j in range(0,n,1024):
    k=order[j:j+1024];p,v,r=m(F[k],OR[k],C[k],L[k],DC[k],DL[k]);loss=(((r-rho[k])/m.rho_res_std)**2).mean()+(((v-Y[k,3:])/m.vres_std)**2).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
   m.eval();er=ev=0.
   with torch.no_grad():
    for j in range(0,len(va),1024):
     k=va[j:j+1024];p,v,_=m(F[k],OR[k],C[k],L[k],DC[k],DL[k]);er+=(p-Y[k,:3]).square().sum().item();ev+=(v-Y[k,3:]).square().sum().item()
   er=(er/len(va))**.5;ev=(ev/len(va))**.5;score=er/c['state_scales'][0]+ev/c['state_scales'][1];row=dict(epoch=ep+1,position_km_rmse=er,velocity_km_s_rmse=ev,common_score=score,seconds=time.time()-start);hist.append(row)
   if score<best:
    best=score;stale=0;torch.save(dict(state=copy.deepcopy(m.state_dict()),width=width,validation=row,config=dict(coordinates='observer',encoder_frame='original inertial',velocity_anchor='range-rate reconstruction' if is_rate else 'direct conditional',range_rate=is_rate,full_observations=full,seed=42,lr=lr,epochs=250,batch_size=1024)),out/(stem+'.pt'))
   else:stale+=1
   (out/(stem+'_history.json')).write_text(json.dumps(hist));print(json.dumps(dict(run=stem,width=width,**row)),flush=True)
   if stale>=30:break
 print('OBSERVER HEAD TRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
