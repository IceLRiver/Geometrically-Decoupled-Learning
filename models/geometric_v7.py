"""Observation-only current GDL/TimeFormer API."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/v7'))
from inference_v7 import selected,predict
def load_model(dataset='public',method='GDL',device='cpu'):
 return selected(dataset,method,device)
def predict_states(bundle,X):
 X=np.asarray(X)
 if X.ndim!=3 or X.shape[1:]!=(40,6):raise ValueError('Expected observation array (N,40,6).')
 return predict(bundle,X)
