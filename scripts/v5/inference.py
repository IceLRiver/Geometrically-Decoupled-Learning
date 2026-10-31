"""Numerically stable observation-only inference for the frozen model parameters."""
import numpy as np
import torch
from stable_inference import load_checkpoint,predict,load_recent,predict_recent,load_corrected,predict_corrected
def load_model(path,device=None):
 device=device or ('cuda' if torch.cuda.is_available() else 'cpu');c=torch.load(path,map_location='cpu')
 if 'rho_res_std' in c['state']:return {'kind':'corrected','bundle':load_corrected(path,device=device)}
 if 'method' in c:return {'kind':'recent','bundle':load_recent(path,device=device)}
 m,cfg=load_checkpoint(path,device);return {'kind':'primary','model':m,'config':cfg}
def predict_states(bundle,X):
 X=np.asarray(X)
 if X.ndim!=3 or X.shape[1:]!=(40,6):raise ValueError('Expected observations (N,40,6).')
 if bundle['kind']=='corrected':return predict_corrected(bundle['bundle'],X)
 if bundle['kind']=='recent':return predict_recent(bundle['bundle'],X)
 return predict(bundle['model'],bundle['config'],X)
