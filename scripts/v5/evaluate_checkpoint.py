from pathlib import Path
import argparse,json
import numpy as np
from data_v5 import BASE,ROOT,load_data,protocol
from inference import load_model,predict_states
from neural_predict import metrics
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['public','simulation','legacy','real'],required=True);ap.add_argument('--method',choices=['GDL','B-spline-KAN-2025','RGA-KAN-2026'],default='GDL');ap.add_argument('--checkpoint');ap.add_argument('--device');ap.add_argument('--full-test',action='store_true');a=ap.parse_args();X,y,_,idx=load_data(a.dataset)
 spec=json.loads((BASE/'evidence/v5/selected_models.json').read_text())['datasets'][a.dataset]['methods'][a.method]
 if not a.checkpoint and 'checkpoint' not in spec:raise ValueError('Supply --checkpoint for a model trained on your private observations.')
 path=Path(a.checkpoint) if a.checkpoint else BASE/spec['checkpoint'];ids=idx['test'] if a.full_test else np.array(protocol()['datasets'][a.dataset]['test_primary']);bundle=load_model(path,a.device);state=predict_states(bundle,X[ids]);np.savez_compressed(ROOT/'evidence'/f'{a.dataset}__{a.method}.npz',indices=ids,state=state);print(json.dumps(metrics(state,y[ids].astype(float)),indent=2))
