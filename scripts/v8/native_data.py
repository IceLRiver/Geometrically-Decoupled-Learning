from paths import BASE
import json,numpy as np
from data_v5 import load_data

def load_native(dataset):
 if dataset not in ['public','legacy']:raise ValueError('Only public and simulation observations are released.')
 X,y,meta,idx=load_data(dataset);folder=BASE/'data/native_lengths';m=json.loads((folder/(dataset+'_manifest.json')).read_text());prefix=[];mask=[]
 for row in m['chunks']:
  with np.load(folder/row['file']) as z:prefix.append(z['prefix20']);mask.append(z['valid60'])
 return dict(X60=np.concatenate([np.concatenate(prefix),X.astype(float)],axis=1),y=y,valid60=np.concatenate(mask),**idx)
