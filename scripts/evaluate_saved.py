from pathlib import Path
import argparse,json,numpy as np,torch
from run_learning import Net,evaluate,ROOT
ap=argparse.ArgumentParser();ap.add_argument('--method',default='Geometric-KAN');ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
path=ROOT/'evidence/synthetic_checkpoints'/f'{args.method}_{args.seed}.pt'
state=torch.load(path,map_location='cpu',weights_only=True)
keys=[f'{name}_{stat}' for name in ['rho','dr','pos','vel'] for stat in ['mean','std']]
model=Net(args.method,{k:state[k].numpy() for k in keys});model.load_state_dict(state)
data=np.load(ROOT/'evidence/synthetic_arcs.npz');split=json.loads((ROOT/'evidence/synthetic_split.json').read_text());idx=split['test_indices']
metrics,_,_=evaluate(model,torch.tensor(data['X'][idx],dtype=torch.float32),data['y'][idx])
print(json.dumps(metrics,indent=2))
