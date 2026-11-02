from pathlib import Path
import json
import numpy as np
import torch
from duration_features import R,B,extract,velocity_design
from native_derivatives import derivatives
from head_utils import extras,apply
def predict(ds,method,N,X,device='cpu'):
 assert N in [10,20,60]
 spec=json.loads((B/'evidence/v8/models.json').read_text())['duration'][ds][str(N)][method];ck=torch.load(B/spec['checkpoint'],map_location='cpu');c=extract(ds,method,X,N,device=device);ex,_=extras(c,ck['cross'],ck['extra_stats'])
 if method=='GDL':
  rho=np.logaddexp(0,apply(ck['range'],np.c_[c['phi'],ex]));p=c['cam']+rho*c['los'];vp=np.c_[velocity_design(ds,c['h'],p),ex];dc,dl=c['dc'],c['dl']
  if 'anchor_degree' in ck:dc,dl=derivatives(X[:,-N:],c['B'],ck['anchor_degree'])
  if ck.get('append_anchor_features',False):vp=np.c_[vp,dc/8,dl/.001]
  residual=apply(ck['velocity'],vp);v=dc+rho*dl+(residual*c['los'] if ck['range_rate'] else residual);state=np.c_[p,v]
 else:state=apply(ck['head'],np.c_[c['phi'],ex])
 return np.c_[np.einsum('nbi,nb->ni',c['B'],state[:,:3]),np.einsum('nbi,nb->ni',c['B'],state[:,3:])]
