"""Observation-only multi-length GDL/representation API."""
from pathlib import Path
import sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/v8'))
from paths import BASE
from duration_inference import predict as predict_adapted
from inference_v7 import selected,predict as predict_anchor
from stable_inference import load_recent,predict_recent

def predict_states(dataset,method,X,measurements=40,device='cpu'):
 X=np.asarray(X)
 if dataset not in ['public','legacy']:raise ValueError('Released checkpoints cover public and simulation sources.')
 if measurements not in [10,20,40,60] or X.ndim!=3 or X.shape[2]!=6 or X.shape[1]<measurements:raise ValueError('Expected N native observations, with six values per sample.')
 if measurements!=40:return predict_adapted(dataset,method,measurements,X,device=device)
 if method in ['GDL','TimeFormer-2026']:return predict_anchor(selected(dataset,method,device),X[:,-40:])
 return predict_recent(load_recent(BASE/'checkpoints/v5'/f'{dataset}__{method}.pt',device),X[:,-40:])
