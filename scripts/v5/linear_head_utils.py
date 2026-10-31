import torch
import numpy as np
def design(layer,x):
 gate=torch.tanh(layer.res_gate);hidden=layer.res_mlp[1](layer.res_mlp[0](x.unsqueeze(-1)));hidden=(hidden*gate[None,:,None]).sum(1)
 return torch.cat([x,torch.sin(x*layer.omega),torch.cos(x*layer.omega),hidden],1)

def fit_head(layer,f,target,train,val,truth_scale,output_mean,output_std):
 tr=f[train].double();va=f[val].double();mu=tr.mean(0);std=tr.std(0).clamp_min(1e-7);z=(tr-mu)/std;zv=(va-mu)/std;t=(target[train].double()-output_mean)/output_std;tv=target[val].double();tm=t.mean(0);A=z.T@z/len(z);b=z.T@(t-tm)/len(z);records=[];best=None
 for ridge in [1e-12,1e-10,1e-8,1e-6,1e-4,.01,1.]:
  w=torch.linalg.solve(A+ridge*torch.eye(A.shape[0],device=A.device),b);pred=(zv@w+tm)*output_std+output_mean;rmse=torch.sqrt(torch.mean(torch.sum((pred-tv)**2,dim=1))).item();rec=dict(ridge=ridge,validation_rmse=rmse);records.append(rec)
  if best is None or rmse<best[0]:best=(rmse,w/std[:,None],tm-(mu/std)@w,rec)
 w=best[1].T.float();bias=best[2].float();ni=layer.in_dim
 with torch.no_grad():
  layer.w_x.copy_(w[:,:ni]);layer.w_sin.copy_(w[:,ni:2*ni]);layer.w_cos.copy_(w[:,2*ni:3*ni]);layer.res_mlp[2].weight.copy_(w[:,3*ni:]);layer.res_mlp[2].bias.zero_();layer.bias.copy_(bias)
 return dict(selected=best[3],search=records)

