"""Executable, source-traceable learning benchmark. No surrogate predictions.
Train/validation/test are target-disjoint. Test labels are used only in evaluation.
"""
from pathlib import Path
import argparse, csv, json, time, hashlib
import numpy as np
import torch
from torch import nn
from original_models import PhysKANLayer

ROOT=Path(__file__).resolve().parents[1]
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)

class Net(nn.Module):
    def __init__(self,variant,stats):
        super().__init__(); self.variant=variant
        for k,v in stats.items(): self.register_buffer(k,torch.tensor(v,dtype=torch.float32))
        self.decoupled=variant not in ['Direct-KAN','Direct-MLP']
        self.triplet=variant.startswith('RangeRate')
        self.kan=variant not in ['Direct-MLP','Geometric-MLP'] and not self.triplet
        d=32
        def layer(i,o):
            if self.kan: return PhysKANLayer(i,o,16,variant!='FixedFrequency-KAN',1.,True)
            return nn.Sequential(nn.Linear(i,d),nn.Tanh(),nn.Linear(d,o))
        if self.triplet:
            # 2026 range/range-rate concept adapted to satellite observer geometry.
            self.regressor=nn.Sequential(nn.Linear(18,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2))
        else:
            self.embed=layer(126,d); self.mid=layer(d,d); self.norm=nn.LayerNorm(d)
            self.score=nn.Sequential(nn.Linear(d,16),nn.GELU(),nn.Linear(16,1))
            self.range=layer(d,1); self.vel=layer(d+3,3); self.state=layer(d,6)
    def forward(self,x):
        z=x.clone(); z[:,:,:3]=z[:,:,:3]/42000.
        if self.triplet:
            out=self.regressor(z[:,[0,19,39]].flatten(1))
            rho=self.rho_mean+self.rho_std*out[:,:1]
            dr=self.dr_mean+self.dr_std*out[:,1:2]
            # Least-squares LOS/observer derivatives over the complete arc.
            tau=torch.arange(40,device=x.device,dtype=x.dtype)-19.5
            dl=(x[:,:,3:]*tau[None,:,None]).sum(1)/(tau*tau).sum()
            dc=(x[:,:,:3]*tau[None,:,None]).sum(1)/(tau*tau).sum()
            los=x[:,-1,3:]; dl=dl-(dl*los).sum(1,keepdim=True)*los
            pos=x[:,-1,:3]+rho*los
            vel=dc+dr*los+rho*dl
        else:
            e=torch.cat([z,z*z,z**3,torch.sin(2*np.pi*z),torch.cos(2*np.pi*z),torch.log(z.abs()+1e-8),torch.exp(z)],dim=-1)
            c=torch.cat([e[:,0:1].expand(-1,38,-1),e[:,1:39],e[:,39:40].expand(-1,38,-1)],dim=-1)
            h=self.embed(c.reshape(-1,126)); h=self.norm(2*h+self.mid(h)).reshape(-1,38,32)
            h=(h*torch.softmax(self.score(h),dim=1)).sum(1)
            if self.decoupled:
                rho=self.rho_mean+self.rho_std*self.range(h)
                pos=x[:,-1,:3]+rho*x[:,-1,3:]
                vel=self.vel_mean+self.vel_std*self.vel(torch.cat([h,pos/42000.],dim=1))
            else:
                out=self.state(h); pos=self.pos_mean+self.pos_std*out[:,:3]
                vel=self.vel_mean+self.vel_std*out[:,3:]
                rho=((pos-x[:,-1,:3])*x[:,-1,3:]).sum(1,keepdim=True)
        return pos,vel,rho

def evaluate(model,X,y):
    model.eval(); pp=[]; vv=[]; rr=[]
    with torch.no_grad():
        for j in range(0,len(X),64):
            p,v,r=model(X[j:j+64]); pp.append(p.cpu().numpy()); vv.append(v.cpu().numpy()); rr.append(r.cpu().numpy())
    p=np.concatenate(pp); v=np.concatenate(vv); r=np.concatenate(rr).ravel()
    er=np.linalg.norm(p-y[:,:3],axis=1); ev=np.linalg.norm(v-y[:,3:],axis=1)
    finite=np.isfinite(p).all(1)&np.isfinite(v).all(1)
    stats={'n':len(y),'finite_fraction':float(finite.mean()),'positive_range_fraction':float((r>0).mean())}
    for key,a in [('position_km',er),('velocity_km_s',ev)]:
        for metric,value in [('mean',np.mean(a)),('rmse',np.sqrt(np.mean(a*a))),('median',np.median(a)),('p95',np.quantile(a,.95))]: stats[key+'_'+metric]=float(value)
    return stats,p,v

def load_observations(dataset):
    if dataset=='synthetic':
        path=ROOT/'evidence/synthetic_arcs.npz'
        data=np.load(path);X=data['X'].astype('float32');y=data['y'].astype('float32')
        meta=json.loads((ROOT/'evidence/synthetic_arcs_meta.json').read_text(encoding='utf-8'))
        return X,y,meta,hashlib.sha256(path.read_bytes()).hexdigest()
    folder=ROOT/'data/multiregime';path=folder/'manifest.json';manifest=json.loads(path.read_text())
    inputs=[];targets=[];meta=[]
    for chunk in manifest['chunks']:
        with np.load(folder/chunk['file']) as data:
            inputs.append(data['X']);targets.append(data['y'])
            meta.extend({'target':int(oid),'split':['train','val','test'][int(code)]} for oid,code in zip(data['orbit_id'],data['split']))
    return np.concatenate(inputs),np.concatenate(targets),meta,hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['synthetic','coverage'],default='synthetic'); ap.add_argument('--epochs',type=int,default=120); args=ap.parse_args()
    X,y,meta,input_hash=load_observations(args.dataset)
    tr=np.flatnonzero([m['split']=='train' for m in meta]); va=np.flatnonzero([m['split']=='val' for m in meta]); te=np.flatnonzero([m['split']=='test' for m in meta])
    assert not (set(meta[i]['target'] for i in tr)&set(meta[i]['target'] for i in te))
    manifest={'dataset':args.dataset,'train_indices':tr.tolist(),'validation_indices':va.tolist(),'test_indices':te.tolist(),'seeds':[42,43,44], 'epochs_max':args.epochs,'patience':25,'batch_size':64,'learning_rate':.001,'optimizer':'AdamW','device':DEVICE,'hardware':torch.cuda.get_device_name() if DEVICE=='cuda' else 'CPU','torch_version':torch.__version__,'input_sha256':input_hash}
    (ROOT/'evidence'/f'{args.dataset}_split.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    rho=np.linalg.norm(y[:,:3]-X[:,-1,:3],axis=1,keepdims=True)
    camv=(X[:,-1,:3]-X[:,-2,:3]); dr=((y[:,3:]-camv)*X[:,-1,3:]).sum(1,keepdims=True)
    stats={}
    for name,a in [('rho',rho),('dr',dr),('pos',y[:,:3]),('vel',y[:,3:])]:
        stats[name+'_mean']=a[tr].mean(0); stats[name+'_std']=np.maximum(a[tr].std(0),.001)
    xt=torch.from_numpy(X).to(DEVICE); yt=torch.from_numpy(y).to(DEVICE); rt=torch.from_numpy(rho).to(DEVICE)
    results=[]; outdir=ROOT/'evidence'/f'{args.dataset}_checkpoints'; outdir.mkdir(exist_ok=True)
    variants=['Geometric-KAN','Direct-KAN','Geometric-MLP','Direct-MLP','NoRangeSupervision-KAN','FixedFrequency-KAN','RangeRate-MLP-2026-adapted','RangeRate-Physics-2026-adapted']
    for variant in variants:
        for seed in manifest['seeds']:
            torch.manual_seed(seed); np.random.seed(seed)
            model=Net(variant,stats).to(DEVICE); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
            best=float('inf'); stale=0; best_epoch=0; best_state=None; begin=time.perf_counter(); history=[]
            for epoch in range(args.epochs):
                model.train(); accum=[]
                _ = np.random.permutation(tr)  # retain the recorded RNG stream
                perm=np.random.permutation(tr)
                for j in range(0,len(perm),64):
                    ids=perm[j:j+64]; p,v,r=model(xt[ids]); loss=((v-yt[ids,3:])/model.vel_std).square().mean()
                    if variant in ['Direct-KAN','Direct-MLP']: loss=loss+((p-yt[ids,:3])/model.pos_std).square().mean()
                    elif variant!='NoRangeSupervision-KAN': loss=loss+((r-rt[ids])/model.rho_std).square().mean()
                    if variant=='RangeRate-Physics-2026-adapted':
                        energy=.5*v.square().sum(1)-398600.4418/p.norm(dim=1).clamp_min(1.)
                        loss=loss+torch.relu(energy).square().mean()/100.
                    optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step(); accum.append(float(loss.detach()))
                val,_,_=evaluate(model,xt[va],y[va]); score=val['position_km_rmse']/float(stats['rho_std'][0])+val['velocity_km_s_rmse']/float(np.linalg.norm(stats['vel_std']))
                history.append({'epoch':epoch+1,'train_loss':float(np.mean(accum)),'validation_score':score})
                if np.isfinite(score) and score<best:
                    best=score; stale=0; best_epoch=epoch+1; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else: stale+=1
                if stale>=25: break
            if best_state is None: raise RuntimeError('No finite checkpoint')
            model.load_state_dict(best_state); res,p,v=evaluate(model,xt[te],y[te]); res.update(method=variant,seed=seed,best_epoch=best_epoch,training_seconds=time.perf_counter()-begin,parameters=sum(z.numel() for z in model.parameters()))
            results.append(res); stem=f'{variant}_{seed}'; torch.save(best_state,outdir/(stem+'.pt'))
            np.savez_compressed(outdir/(stem+'_predictions.npz'),indices=te,position=p,velocity=v)
            (outdir/(stem+'_history.json')).write_text(json.dumps(history),encoding='utf-8')
            (ROOT/'evidence'/f'{args.dataset}_learning_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(json.dumps(res),flush=True)

if __name__=='__main__': main()
