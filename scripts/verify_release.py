"""Verify released checksums, target partitions and saved prediction metrics."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/v4'))
from benchmark_data import load_data,freeze_protocol

def main():
 manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
 for item in manifest['files']:
  p=ROOT/item['file'];assert p.stat().st_size==item['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
 rows=json.loads((ROOT/'evidence/v4/results.json').read_text());count=0
 for ds,n in [('public',9632),('simulation',120000)]:
  X,y,meta,idx=load_data(ds);assert X.shape==(n,40,6) and y.shape==(n,6)
  assert np.isfinite(X).all() and np.isfinite(y).all()
  assert np.allclose(np.linalg.norm(X[:,:,3:],axis=-1),1,atol=2e-7)
  ids=freeze_protocol()['datasets'][ds]['test_primary']
  for row in [r for r in rows if r['dataset']==ds]:
   with np.load(ROOT/'evidence/v4/predictions'/f"{ds}__{row['method']}.npz") as z:
    assert z['indices'].tolist()==ids
    state=z['state'] if 'state' in z else np.c_[z['position'],z['velocity']]
    assert np.isfinite(state).all()
    for cols,key in [(slice(0,3),'position_km_rmse'),(slice(3,6),'velocity_km_s_rmse')]:
     value=np.sqrt(np.mean(np.sum((state[:,cols].astype(float)-y[z['indices'],cols].astype(float))**2,axis=1)))
     assert np.isclose(value,row[key],rtol=1e-9,atol=1e-9),(ds,row['method'],key,value,row[key])
    count+=1
 print(f'PASS: all release checksums, target-disjoint datasets and {count} saved-prediction result rows.')
if __name__=='__main__':main()
