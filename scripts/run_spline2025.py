"""Explicit satellite-task adaptation of Liu et al. (2025), DOI 10.11834/jig.250147.
Original protocol is 300 s / six samples / initial state; this adaptation uses
six samples from the common 39 s arc and predicts the terminal state.
Widths/grid and training budget below are adaptation choices, not a reproduction.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from run_learning import ROOT,DEVICE,evaluate

class SplineLayer(nn.Module):
    def __init__(self,i,o):
        super().__init__(); self.base=nn.Linear(i,o)
        self.coef=nn.Parameter(torch.randn(o,i,8)*.01)
        self.register_buffer('knots',torch.arange(-3,9,dtype=torch.float32)*.4-1)
    def forward(self,x):
        t=self.knots
        b=((x[:,:,None]>=t[:-1])&(x[:,:,None]<t[1:])).to(x.dtype)
        for k in range(1,4):
            b=(x[:,:,None]-t[:-(k+1)])/(t[k:-1]-t[:-(k+1)])*b[:,:,:-1]+(t[k+1:]-x[:,:,None])/(t[k+1:]-t[1:-k])*b[:,:,1:]
        return self.base(F.silu(x))+F.linear(b.flatten(1),self.coef.flatten(1))

class SplineNet(nn.Module):
    def __init__(self,low,high,mean,std):
        super().__init__()
        for k,v in dict(low=low,high=high,mean=mean,std=std).items():self.register_buffer(k,torch.tensor(v,dtype=torch.float32))
        dims=[60,64,128,64,32,6];self.layers=nn.ModuleList([SplineLayer(i,o) for i,o in zip(dims[:-1],dims[1:])])
    def forward(self,z):
        h=2*(z-self.low)/(self.high-self.low).clamp_min(1e-6)-1
        for layer in self.layers:h=layer(h)
        return h*self.std+self.mean

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['synthetic'],default='synthetic');args=ap.parse_args()
    ds=args.dataset;data=np.load(ROOT/'evidence'/f'{ds}_arcs.npz');X=data['X'];y=data['y'].astype('float32')
    split=json.loads((ROOT/'evidence'/f'{ds}_split.json').read_text());tr=np.array(split['train_indices']);va=np.array(split['validation_indices']);te=np.array(split['test_indices'])
    los=X[:,:,3:];az=np.unwrap(np.arctan2(los[:,:,1],los[:,:,0]),axis=1);el=np.arcsin(np.clip(los[:,:,2],-1,1))
    a=np.stack([az,el],-1);da=np.gradient(a,axis=1);cam=X[:,:,:3];cv=np.gradient(cam,axis=1)
    z=np.concatenate([a,da,cam,cv],-1)[:,[0,8,16,23,31,39]].reshape(-1,60).astype('float32')
    low=z[tr].min(0);high=z[tr].max(0);mean=y[tr].mean(0);std=np.maximum(y[tr].std(0),.001)
    zt=torch.tensor(z,device=DEVICE);yt=torch.tensor(y,device=DEVICE);out=[]
    outdir=ROOT/'evidence'/f'{ds}_checkpoints';outdir.mkdir(exist_ok=True)
    for seed in [42,43,44]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=SplineNet(low,high,mean,std).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        best=float('inf');state=None;stale=0;begin=time.perf_counter()
        for epoch in range(120):
            model.train();perm=rng.permutation(tr)
            for j in range(0,len(tr),64):
                ids=perm[j:j+64];pred=model(zt[ids]);loss=((pred[:,:3]-yt[ids,:3])/1000).norm(dim=1).mean()+10*(pred[:,3:]-yt[ids,3:]).norm(dim=1).mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            model.eval()
            with torch.no_grad():
                pred=model(zt[va]);score=float(((pred[:,:3]-yt[va,:3])/1000).norm(dim=1).mean()+10*(pred[:,3:]-yt[va,3:]).norm(dim=1).mean())
            if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1;stale=0
            else:stale+=1
            if stale>=25:break
        model.load_state_dict(state);model.eval()
        with torch.no_grad():pred=model(zt[te]).cpu().numpy()
        er=np.linalg.norm(pred[:,:3]-y[te,:3],axis=1);ev=np.linalg.norm(pred[:,3:]-y[te,3:],axis=1)
        res=dict(method='SplineKAN-2025-adapted',seed=seed,n=len(te),best_epoch=best_epoch,parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-begin,finite_fraction=float(np.isfinite(pred).all(1).mean()))
        for label,a in [('position_km',er),('velocity_km_s',ev)]:
            res.update({label+'_mean':float(a.mean()),label+'_rmse':float(np.sqrt(np.mean(a*a))),label+'_median':float(np.median(a)),label+'_p95':float(np.quantile(a,.95))})
        out.append(res);stem=f'SplineKAN-2025-adapted_{seed}';torch.save(state,outdir/(stem+'.pt'));np.savez_compressed(outdir/(stem+'_predictions.npz'),indices=te,position=pred[:,:3],velocity=pred[:,3:])
        (ROOT/'evidence'/f'{ds}_spline2025_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(res),flush=True)
if __name__=='__main__':main()
