from paths import BASE
import numpy as np
from scipy.spatial.distance import cdist
from rbf_baseline import representation_data,inverse_state
def rbf(ds,X):
 p=BASE/'checkpoints/v5/legacy__RBFNN-2024.npz' if ds=='legacy' else BASE/'checkpoints/v4'/('public__RBFNN-2024.npz');c=dict(np.load(p));f,_,transform=representation_data(X,np.zeros((len(X),6)),str(c.get('representation','inertial')));f=(f-c['mu'])/c['std'];A=np.c_[np.exp(-cdist(f,c['C'],'sqeuclidean')/c['width']),np.ones(len(f))];return inverse_state(A@c['W']*c['ys']+c['ym'],transform,np.arange(len(X)))