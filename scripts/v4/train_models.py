"""Matched-input neural comparisons. Hyperparameter search reads validation only."""
from pathlib import Path
import sys,json,time,argparse,copy
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fast_kan import FastPhysKANLayer as PhysKANLayer
from benchmark_data import load_data,kinematics,freeze_protocol,ROOT,observer_frame
torch.set_num_threads(4);DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
class Model(nn.Module):
 def __init__(self,kind,stats,width=64,depth=2,temporal_stride=1):
  super().__init__();self.kind=kind;self.temporal_stride=temporal_stride
  for k,v in stats.items():self.register_buffer(k,torch.tensor(v,dtype=torch.float32))
  kan='MLP' not in kind
  def layer(i,o):return PhysKANLayer(i,o,16,True,1.,True,True) if kan else nn.Sequential(nn.Linear(i,width),nn.SiLU(),nn.Linear(width,o))
  self.embed=layer(126,width);self.blocks=nn.ModuleList([layer(width,width) for _ in range(depth)]);self.norm=nn.LayerNorm(width);self.score=nn.Linear(width,1)
  self.feature=nn.Sequential(nn.Linear(len(stats['f_mean']),width),nn.SiLU(),nn.Linear(width,width))
  self.range=layer(width,1);self.velocity=layer(width+3,3);self.state=layer(width,6)
  if kind.startswith('FixedFrequency'):
   for name,param in self.named_parameters():
    if name.endswith('omega'):param.requires_grad_(False)
 def forward(self,x,f,dc,dl):
  z=x.clone();z[:,:,:3]/=42000
  e=torch.cat([z,z*z,z**3,torch.sin(2*np.pi*z),torch.cos(2*np.pi*z),torch.log(z.abs()+1e-8),torch.exp(z)],-1)
  c=torch.cat([e[:,:1].expand(-1,38,-1),e[:,1:39],e[:,39:].expand(-1,38,-1)],-1)
  if self.temporal_stride>1:
   ids=torch.arange(0,38,self.temporal_stride,device=c.device)
   if ids[-1]!=37:ids=torch.cat([ids,ids.new_tensor([37])])
   c=c[:,ids]
  count=c.shape[1]
  h=self.embed(c.reshape(-1,126))
  for block in self.blocks:h=self.norm(h+.5*block(h))
  h=h.reshape(-1,count,h.shape[-1]);h=(h*torch.softmax(self.score(h),1)).sum(1)+self.feature(((f-self.f_mean)/self.f_std).clamp(-10,10))
  if self.kind.startswith('Direct'):
   raw=self.state(h);p=self.p_mean+self.p_std*raw[:,:3];v=self.v_mean+self.v_std*raw[:,3:];rho=((p-x[:,-1,:3])*x[:,-1,3:]).sum(-1,keepdim=True)
  else:
   if self.kind.startswith('Prior'):
    R=x[:,-1,:3];L=x[:,-1,3:];dot=(R*L).sum(-1,keepdim=True);inside=dot*dot+self.radius_mean**2-(R*R).sum(-1,keepdim=True)
    anchor=(-dot+inside.clamp(min=1).sqrt()).clamp(min=1)
    rho=anchor*torch.exp(.1*self.range(h).clamp(-30,30))
   else:rho=torch.nn.functional.softplus(self.range(h))*self.rho_scale
   p=x[:,-1,:3]+rho*x[:,-1,3:]
   raw=self.velocity(torch.cat([h,p/42000],-1))
   if self.kind.startswith('Original'):v=self.v_mean+self.v_std*raw
   else:
    base=dc+rho*dl
    if self.kind.startswith('Prior'):
     denom=(p*x[:,-1,3:]).sum(-1,keepdim=True);safe=torch.where(denom.abs()>1,denom,torch.ones_like(denom));radial=-(p*base).sum(-1,keepdim=True)/safe;base=base+radial*x[:,-1,3:]
    v=base+self.res_mean+self.res_std*raw
  return p,v,rho
def tensors(X,y,idx,degree,kind='Geometric-KAN',extra_features=None):
 f,dc,dl=kinematics(X,degree);rho=np.linalg.norm(y[:,:3]-X[:,-1,:3],axis=1,keepdims=True);base=dc+rho*dl
 if kind.startswith('Prior'):
  den=np.sum(y[:,:3]*X[:,-1,3:],axis=1,keepdims=True);safe=np.where(np.abs(den)>1,den,1);base-=np.sum(y[:,:3]*base,axis=1,keepdims=True)/safe*X[:,-1,3:]
 if extra_features is not None:f=np.c_[f,extra_features].astype('float32')
 res=y[:,3:]-base;tr=idx['train'];stats={'radius_mean':np.array([np.linalg.norm(y[tr,:3],axis=1).mean()],np.float32)}
 for key,z in [('f',f),('p',y[:,:3]),('v',y[:,3:]),('res',res),('rho',rho)]:stats[key+'_mean']=z[tr].mean(0);stats[key+'_std']=np.maximum(z[tr].std(0),1e-3)
 stats['rho_scale']=np.array([np.median(rho[tr])],np.float32)
 return [torch.tensor(z,device=DEVICE,dtype=torch.float32) for z in [X,f,dc,dl,y,rho]],stats
def metrics(model,T,ids):
 ps=[];vs=[];model.eval()
 with torch.no_grad():
  for j in range(0,len(ids),256):
   k=ids[j:j+256];p,v,_=model(*[a[k] for a in T[:4]]);ps.append(p.cpu().numpy());vs.append(v.cpu().numpy())
 p=np.concatenate(ps);v=np.concatenate(vs);truth=T[4][ids].cpu().numpy();er=np.linalg.norm(p-truth[:,:3],axis=1);ev=np.linalg.norm(v-truth[:,3:],axis=1)
 return {'position_km_rmse':float(np.sqrt(np.mean(er**2))),'velocity_km_s_rmse':float(np.sqrt(np.mean(ev**2)))},p,v
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--kind',default='Geometric-KAN');ap.add_argument('--width',type=int,default=64);ap.add_argument('--depth',type=int,default=2);ap.add_argument('--degree',type=int,default=2);ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--velocity-scale',choices=['orbital','residual'],default='residual');ap.add_argument('--weight-decay',type=float,default=.0001);ap.add_argument('--coordinates',choices=['inertial','observer','hybrid'],default='inertial');ap.add_argument('--batch-size',type=int,default=128);ap.add_argument('--range-weight',type=float,default=1.);ap.add_argument('--final',action='store_true');a=ap.parse_args()
 protocol=freeze_protocol();X,y,meta,idx=load_data(a.dataset)
 extra=None
 if a.coordinates=='hybrid':
  _,dv,_=kinematics(X);extra=np.c_[X[:,0,:3]/42000,dv/8]
 if a.coordinates in ['observer','hybrid']:X,y,B=observer_frame(X,y)
 T,stats=tensors(X,y,idx,a.degree,a.kind,extra);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
 model=Model(a.kind,stats,a.width,a.depth).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
 scale=model.res_std if a.velocity_scale=='residual' else model.v_std
 out=ROOT/'evidence/neural_v4';out.mkdir(exist_ok=True);stem=f'{a.dataset}_{a.kind}_w{a.width}d{a.depth}_p{a.degree}_lr{a.lr}_wd{a.weight_decay}_{a.velocity_scale}_s{a.seed}'+('_'+a.coordinates if a.coordinates!='inertial' else '')+(f'_b{a.batch_size}' if a.batch_size!=128 else '')+(f'_lambda{a.range_weight}' if a.range_weight!=1 else '');best=np.inf;stale=0;history=[];begin=time.time()
 for epoch in range(a.epochs):
  model.train();perm=rng.permutation(idx['train']);losses=[]
  for j in range(0,len(perm),a.batch_size):
   k=perm[j:j+a.batch_size];p,v,rho=model(*[z[k] for z in T[:4]]);loss=(((v-T[4][k,3:])/scale)**2).mean()
   loss+=(((p-T[4][k,:3])/model.p_std)**2).mean() if a.kind.startswith('Direct') else a.range_weight*(((rho-T[5][k])/model.rho_std)**2).mean()
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(loss.detach().item())
  val,_,_=metrics(model,T,idx['val']);score=val['position_km_rmse']/float(stats['rho_std'][0])+val['velocity_km_s_rmse']/float(np.linalg.norm(scale.cpu().numpy()));history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':val,'selection_score':score})
  if score<best:
   best=score;stale=0;state=copy.deepcopy(model.state_dict());torch.save({'state':state,'config':vars(a),'best_epoch':epoch+1,'validation':val,'selection_score':score},out/(stem+'.pt'))
  else:stale+=1
  (out/(stem+'_history.json')).write_text(json.dumps(history),encoding='utf-8')
  print(json.dumps({'run':stem,**history[-1],'elapsed_s':time.time()-begin}),flush=True)
  if stale>=20:break
 if a.final:
  model.load_state_dict(state);ids=np.array(protocol['datasets'][a.dataset]['test_primary']);m,p,v=metrics(model,T,ids)
  if a.coordinates in ['observer','hybrid']:p=np.einsum('nbi,nb->ni',B[ids],p);v=np.einsum('nbi,nb->ni',B[ids],v)
  np.savez_compressed(out/(stem+'_test_predictions.npz'),indices=ids,position=p,velocity=v);(out/(stem+'_test_metrics.json')).write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
