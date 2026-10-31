"""Re-run the fixed component study on the retained historical simulation only.
No test references enter optimization or checkpoint selection.
"""
from pathlib import Path
import sys,json,time,copy,hashlib
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'runs/v6'
sys.path.insert(0,str(REPO/'scripts/v5'))
from data_v5 import load_data,observer_frame,protocol
from train_models import Model,tensors,metrics,DEVICE
import torch
import numpy as np
from torch import nn

OUT=ROOT/'evidence/matched_ablation';OUT.mkdir(parents=True,exist_ok=True)
KINDS=[('Geometric-KAN',1.),('Direct-KAN',1.),('Geometric-MLP',1.),('NoRange-KAN',0.),('FixedFrequency-KAN',1.),('Original-KAN',1.)]
X,y,_,idx=load_data('legacy');X,y,B=observer_frame(X,y)
for kind,lam in KINDS:
    stem='legacy_'+kind
    if (OUT/(stem+'_final.json')).exists():continue
    T,stats=tensors(X,y,idx,2,kind);torch.manual_seed(42);rng=np.random.default_rng(42)
    model=Model(kind,stats,32,1,8).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');stale=0;hist=[];begin=time.time()
    cfg=dict(dataset='legacy',kind=kind,width=32,depth=1,degree=2,coordinates='observer',temporal_stride=8,batch_size=1024,epochs=100,lr=.001,weight_decay=.0001,range_weight=lam,velocity_scale='residual',seed=42)
    for epoch in range(100):
        model.train();perm=rng.permutation(idx['train'])
        for j in range(0,len(perm),1024):
            k=perm[j:j+1024];p,v,rho=model(*[a[k] for a in T[:4]])
            loss=(((v-T[4][k,3:])/model.res_std)**2).mean()
            loss+=(((p-T[4][k,:3])/model.p_std)**2).mean() if kind.startswith('Direct') else lam*(((rho-T[5][k])/model.rho_std)**2).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        val,_,_=metrics(model,T,idx['val']);score=val['position_km_rmse']/float(stats['rho_std'][0])+val['velocity_km_s_rmse']/float(np.linalg.norm(model.res_std.cpu().numpy()))
        row=dict(epoch=epoch+1,validation=val,selection_score=score,elapsed_s=time.time()-begin);hist.append(row)
        if score<best:
            best=score;stale=0;state=copy.deepcopy(model.state_dict());torch.save(dict(state=state,config=cfg,validation=val,selection_score=score,best_epoch=epoch+1),OUT/(stem+'.pt'))
        else:stale+=1
        (OUT/(stem+'_history.json')).write_text(json.dumps(hist));print(json.dumps(dict(run=stem,**row)),flush=True)
        if stale>=20:break
    # Freeze the selected model before accessing the test partition.
    sha=hashlib.sha256((OUT/(stem+'.pt')).read_bytes()).hexdigest();model.load_state_dict(state);model.double().eval()
    ids=np.array(protocol()['datasets']['legacy']['test_primary']);pred=[]
    with torch.no_grad():
        for j in range(0,len(ids),128):
            k=ids[j:j+128];inputs=[torch.tensor(X[k],device=DEVICE,dtype=torch.float64)]+[a[k].double() for a in T[1:4]];p,v,_=model(*inputs);p=p.cpu().numpy();v=v.cpu().numpy()
            pred.append(np.c_[np.einsum('nbi,nb->ni',B[k],p),np.einsum('nbi,nb->ni',B[k],v)])
    z=np.concatenate(pred);np.savez_compressed(OUT/(stem+'_predictions.npz'),indices=ids,state=z)
    _,truth,_,_=load_data('legacy');d=z-truth[ids].astype(float)
    result=dict(kind=kind,checkpoint_sha256=sha,best_epoch=int(np.argmin([h['selection_score'] for h in hist]))+1,position_km_rmse=float(np.sqrt(np.mean(np.sum(d[:,:3]**2,axis=1)))),velocity_km_s_rmse=float(np.sqrt(np.mean(np.sum(d[:,3:]**2,axis=1)))),parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),elapsed_s=time.time()-begin)
    (OUT/(stem+'_final.json')).write_text(json.dumps(result,indent=2));print('FINAL '+json.dumps(result),flush=True)
    del model,opt,T
