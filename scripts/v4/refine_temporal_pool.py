"""Validation-only check and refinement of subsampled temporal combinations.
All 40 measurements still contribute to polynomial/kinematic features.
"""
import json,time,copy,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_models import Model,tensors,metrics,DEVICE
from benchmark_data import ROOT,load_data,observer_frame

def run(path,epochs=60):
 checkpoint=torch.load(path,map_location='cpu');c=dict(checkpoint['config']);c.update(temporal_stride=4,lr=.0003,range_weight=c.get('range_weight',1.),batch_size=2048,epochs=epochs,base_checkpoint_best_epoch=checkpoint['best_epoch'],resume_checkpoint=Path(path).name,training_schedule='validation-selected dense-to-subsampled temporal pooling');X,y,_,idx=load_data(c['dataset']);X,y,_=observer_frame(X,y);T,stats=tensors(X,y,idx,c.get('degree',2),c['kind']);model=Model(c['kind'],stats,c['width'],c['depth'],4).to(DEVICE);model.load_state_dict(checkpoint['state']);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);rng=np.random.default_rng(42);scale=model.res_std;out=ROOT/'evidence/neural_v4';stem=Path(path).stem+'_stride4_refined';history=[];best=np.inf;stale=0;begin=time.time()
 for epoch in range(epochs+1):
  if epoch:
   model.train();perm=rng.permutation(idx['train'])
   for j in range(0,len(perm),2048):
    k=perm[j:j+2048];p,v,rho=model(*[z[k] for z in T[:4]]);loss=(((v-T[4][k,3:])/scale)**2).mean();loss+=(((p-T[4][k,:3])/model.p_std)**2).mean() if c['kind'].startswith('Direct') else c['range_weight']*(((rho-T[5][k])/model.rho_std)**2).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
  val,_,_=metrics(model,T,idx['val']);score=val['position_km_rmse']/float(stats['rho_std'][0])+val['velocity_km_s_rmse']/float(np.linalg.norm(scale.cpu().numpy()));row={'epoch':epoch,'validation':val,'selection_score':score};history.append(row)
  if score<best:
   best=score;stale=0;state=copy.deepcopy(model.state_dict());torch.save({'state':state,'config':c,'validation':val,'selection_score':score,'best_epoch':epoch},out/(stem+'.pt'))
  else:stale+=1
  (out/(stem+'_history.json')).write_text(json.dumps(history));print(json.dumps({'run':stem,**row,'elapsed_s':time.time()-begin}),flush=True)
  if stale>=20:break
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('paths',nargs='+');a=ap.parse_args()
 for p in a.paths:run(p)
