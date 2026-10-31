from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.distance import cdist
from data_v5 import BASE,ROOT,load_data,protocol
from rbf_baseline import representation_data,inverse_state
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['public','simulation','legacy','real'],required=True);ap.add_argument('--checkpoint');a=ap.parse_args();X,y,_,_=load_data(a.dataset);ids=np.array(protocol()['datasets'][a.dataset]['test_primary']);meta=json.loads((BASE/'evidence/v5/selected_models.json').read_text())['datasets'][a.dataset]['methods']['RBFNN-2024']
 if not a.checkpoint and 'checkpoint' not in meta:raise ValueError('Supply a local private-data checkpoint.')
 c=dict(np.load(Path(a.checkpoint) if a.checkpoint else BASE/meta['checkpoint']));f,_,B=representation_data(X[ids],np.zeros((len(ids),6)),str(c['representation'].item()));f=(f-c['mu'])/c['std'];A=np.c_[np.exp(-cdist(f,c['C'],'sqeuclidean')/c['width']),np.ones(len(f))];state=inverse_state(A@c['W']*c['ys']+c['ym'],B,np.arange(len(ids)));np.savez_compressed(ROOT/'evidence'/f'{a.dataset}__RBFNN-2024.npz',indices=ids,state=state);er=np.linalg.norm(state[:,:3]-y[ids,:3].astype(float),axis=1);ev=np.linalg.norm(state[:,3:]-y[ids,3:].astype(float),axis=1);print(json.dumps(dict(position_km_rmse=float(np.sqrt(np.mean(er**2))),velocity_km_s_rmse=float(np.sqrt(np.mean(ev**2)))),indent=2))
