from pathlib import Path
import sys,json
import numpy as np
import torch
BASE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(BASE/'scripts/v5'))
from inference import load_model,predict_states
from data_v5 import load_data
torch.set_num_threads(2)
records=[]
for ds in ['public','legacy']:
    X,_,_,_=load_data(ds)
    for kind in ['Geometric-KAN','Direct-KAN','Geometric-MLP','NoRange-KAN','FixedFrequency-KAN','Original-KAN']:
        stem=f'{ds}_{kind}';z=np.load(BASE/'evidence/v6/predictions'/(stem+'.npz'));ids=z['indices'][:8]
        bundle=load_model(BASE/'checkpoints/v6'/(stem+'.pt'),device='cpu');batch=predict_states(bundle,X[ids]);single=predict_states(bundle,X[ids[:1]])
        error=float(np.max(np.abs(batch-z['state'][:8])));batch_error=float(np.max(np.abs(batch[:1]-single)))
        assert error<1e-6 and batch_error<1e-6,(stem,error,batch_error)
        records.append(dict(dataset=ds,kind=kind,max_absolute_saved_state_difference=error,max_absolute_single_batch_difference=batch_error))
(BASE/'evidence/v6/component_api_checks.json').write_text(json.dumps(records,indent=2))
print('PASS: 12 released component checkpoints reproduce saved states and single/batch inference (CPU, FP64).')
