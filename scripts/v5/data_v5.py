from pathlib import Path
import sys,json,os
import numpy as np
BASE=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('ORBIT_RUN_ROOT',str(BASE/'runs/v5')));(ROOT/'evidence').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(BASE/'scripts/v4'))
from benchmark_data import load_data as load_v4,kinematics,observer_frame

def load_data(ds):
 if ds!='legacy':return load_v4(ds)
 folder=BASE/'data/legacy_simulation';m=json.loads((folder/'manifest.json').read_text());X=[];y=[]
 for entry in m['chunks']:
  with np.load(folder/entry['file']) as z:X.append(z['X'].reshape(-1,40,6));y.append(z['y'])
 X=np.concatenate(X);y=np.concatenate(y);ids=np.arange(len(X));np.random.RandomState(42).shuffle(ids);n=int(.8*len(ids));oldtr=ids[:n];rng=np.random.default_rng(20261101);perm=rng.permutation(oldtr);nv=int(.1*len(oldtr));idx={'train':perm[nv:],'val':perm[:nv],'test':ids[n:]}
 return X,y,None,idx


def protocol():
 return json.loads((BASE/'data/benchmark_protocol_v5.json').read_text())
