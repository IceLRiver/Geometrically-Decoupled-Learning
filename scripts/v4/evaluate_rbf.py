"""Evaluate the validation-selected RBFNN checkpoint using observations only."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.distance import cdist
from benchmark_data import ROOT,BASE,load_data,freeze_protocol
from rbf_baseline import representation_data,inverse_state

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['public','simulation'],required=True);a=ap.parse_args()
 X,y,_,_=load_data(a.dataset);ids=np.array(freeze_protocol()['datasets'][a.dataset]['test_primary'])
 with np.load(BASE/f'checkpoints/v4/{a.dataset}__RBFNN-2024.npz') as z:
  f,_,transform=representation_data(X[ids],np.zeros((len(ids),6)),str(z['representation'].item()));f=(f-z['mu'])/z['std'];A=np.c_[np.exp(-cdist(f,z['C'],'sqeuclidean')/z['width']),np.ones(len(f))];state=inverse_state(A@z['W']*z['ys']+z['ym'],transform,np.arange(len(ids)))
 np.savez_compressed(ROOT/'evidence'/f'{a.dataset}_RBFNN-2024_predictions.npz',indices=ids,state=state)
 print(json.dumps({'position_km_rmse':float(np.sqrt(np.mean(np.sum((state[:,:3]-y[ids,:3])**2,axis=1)))),'velocity_km_s_rmse':float(np.sqrt(np.mean(np.sum((state[:,3:]-y[ids,3:])**2,axis=1))))},indent=2))
if __name__=='__main__':main()
