"""The same training-only affine-head refinement applied to the proposed framework."""
import json,copy
import numpy as np
import torch
from torch import nn
from data_v5 import ROOT,BASE,load_data,kinematics,observer_frame
from neural_predict import load_checkpoint
from linear_head_utils import design
torch.set_num_threads(4)

def basis(layer,x):return layer[1](layer[0](x)) if isinstance(layer,nn.Sequential) else design(layer,x)
def add_delta(layer,w,b):
 with torch.no_grad():
  if isinstance(layer,nn.Sequential):layer[-1].weight.add_(w.T.float().cuda());layer[-1].bias.add_(b.float().cuda())
  else:
   w=w.T.float().cuda();n=layer.in_dim;layer.w_x.add_(w[:,:n]);layer.w_sin.add_(w[:,n:2*n]);layer.w_cos.add_(w[:,2*n:3*n]);layer.res_mlp[2].weight.add_(w[:,3*n:]);layer.bias.add_(b.float().cuda())

def candidates(layer,features,target,n):
 phi=basis(layer,features).detach().double().cpu();old=layer(features).detach().double().cpu();mu=phi[:n].mean(0);std=phi[:n].std(0).clamp_min(1e-7);z=(phi[:n]-mu)/std;t=target[:n].double().cpu()-old[:n];tm=t.mean(0);A=z.T@z/n;B=z.T@(t-tm)/n
 for ridge in [1e-12,1e-10,1e-8,1e-6,1e-4,.01,1.,100.]:
  w=torch.linalg.solve(A+ridge*torch.eye(A.shape[0],dtype=torch.float64),B);yield ridge,w/std[:,None],tm-(mu/std)@w

def run(ds,range_mix=1.):
 old=json.loads((BASE/'evidence/v4/selected_models.json').read_text());spec=old['datasets'][ds]['methods']['GDL'];path=BASE/spec['checkpoint'].replace('\\','/');model,cfg=load_checkpoint(path);X,y,_,idx=load_data(ds);extra=None
 if cfg['coordinates']=='hybrid':_,dv,_=kinematics(X);extra=np.c_[X[:,0,:3]/42000,dv/8]
 X,y,B=observer_frame(X,y);tr=idx['train'];va=idx['val'];ids=np.r_[tr,va];n=len(tr);X=X[ids];Y=torch.tensor(y[ids],device='cuda',dtype=torch.float32);f,dc,dl=kinematics(X)
 if extra is not None:f=np.c_[f,extra[ids]]
 T=[torch.tensor(a,device='cuda',dtype=torch.float32) for a in [X,f,dc,dl]];H=[]
 hook=model.range.register_forward_pre_hook(lambda m,args:H.append(args[0].detach()))
 with torch.no_grad():
  for j in range(0,len(ids),512):model(*[a[j:j+512] for a in T])
 hook.remove();h=torch.cat(H);cam=T[0][:,-1,:3];los=T[0][:,-1,3:];rho=(Y[:,:3]-cam).norm(dim=1,keepdim=True);a=rho/model.rho_scale;target=a+torch.log(-torch.expm1(-a));range_state=copy.deepcopy(model.range.state_dict());best=None;range_records=[]
 with torch.no_grad():
  for ridge,w,b in candidates(model.range,h,target,n):
   model.range.load_state_dict(range_state);add_delta(model.range,w,b);r=model.rho_scale*torch.nn.functional.softplus(model.range(h[n:]));p=cam[n:]+r*los[n:];error=torch.sqrt(torch.mean(torch.sum((p-Y[n:,:3])**2,dim=1))).item();range_records.append(dict(ridge=ridge,validation_position_km_rmse=error))
   if best is None or error<best[0]:best=(error,copy.deepcopy(model.range.state_dict()),ridge)
  model.range.load_state_dict({k:range_state[k]+range_mix*(best[1][k]-range_state[k]) for k in range_state});rhat=model.rho_scale*torch.nn.functional.softplus(model.range(h));p=cam+rhat*los;actual_range_error=torch.sqrt(torch.mean(torch.sum((p[n:]-Y[n:,:3])**2,dim=1))).item();vfeatures=torch.cat([h,p/42000],1);target=(Y[:,3:]-T[2]-rhat*T[3]-model.res_mean)/model.res_std;velocity_state=copy.deepcopy(model.velocity.state_dict());bestv=None;velocity_records=[]
  for ridge,w,b in candidates(model.velocity,vfeatures,target,n):
   model.velocity.load_state_dict(velocity_state);add_delta(model.velocity,w,b);v=T[2][n:]+rhat[n:]*T[3][n:]+model.res_mean+model.res_std*model.velocity(vfeatures[n:]);error=torch.sqrt(torch.mean(torch.sum((v-Y[n:,3:])**2,dim=1))).item();velocity_records.append(dict(ridge=ridge,validation_velocity_km_s_rmse=error))
   if bestv is None or error<bestv[0]:bestv=(error,copy.deepcopy(model.velocity.state_dict()),ridge)
  model.velocity.load_state_dict(bestv[1])
 out=ROOT/'evidence/primary_refitted';out.mkdir(exist_ok=True);c=torch.load(path,map_location='cpu');c['state']=model.state_dict();c['validation_before_refit']=c['validation'];c['validation']=dict(position_km_rmse=actual_range_error,velocity_km_s_rmse=bestv[0]);c['config']['training_schedule']=c['config'].get('training_schedule','AdamW')+' + training-only affine head refit';c['head_refit']=dict(range_search=range_records,velocity_search=velocity_records,range_ridge=best[2],velocity_ridge=bestv[2],range_mix=range_mix);suffix='' if range_mix==1 else '_mix'+str(range_mix).replace('.','p');dest=out/f'{ds}__GDL{suffix}.pt';torch.save(c,dest);dest.with_suffix('.json').write_text(json.dumps(dict(validation=c['validation'],head_refit=c['head_refit']),indent=2));print(ds,range_mix,c['validation'],flush=True)
if __name__=='__main__':
 for ds in ['public','simulation']:run(ds)
