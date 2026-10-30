"""Check data splits and independently recompute every released compact metric."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 data=np.load(ROOT/'evidence/synthetic_arcs.npz'); X,y=data['X'],data['y']
 meta=json.loads((ROOT/'evidence/synthetic_arcs_meta.json').read_text());split=json.loads((ROOT/'evidence/synthetic_split.json').read_text())
 assert X.shape==(2250,40,6) and y.shape==(2250,6)
 assert np.isfinite(X).all() and np.isfinite(y).all()
 assert np.allclose(np.linalg.norm(X[:,:,3:],axis=-1),1.,atol=1e-7)
 ids=[split[k] for k in ['train_indices','validation_indices','test_indices']]
 assert [len(z) for z in ids]==[1554,228,468]
 groups=[set(meta[i]['target'] for i in z) for z in ids]
 assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
 assert sorted(sum(ids,[]))==list(range(len(X)))
 rows=[]
 for name in ['synthetic_learning_results.json','synthetic_spline2025_results.json']:rows+=json.loads((ROOT/'evidence'/name).read_text())
 for row in rows:
  p=ROOT/'evidence/synthetic_checkpoints'/f"{row['method']}_{row['seed']}_predictions.npz";pred=np.load(p);idx=pred['indices']
  assert set(idx)==set(ids[2])
  for k,key,target in [('position','position_km_rmse',y[idx,:3]),('velocity','velocity_km_s_rmse',y[idx,3:])]:
   rmse=np.sqrt(np.mean(np.sum((pred[k].astype(float)-target)**2,axis=1)))
   assert np.isclose(rmse,row[key],rtol=1e-5,atol=1e-3),(p,key,rmse,row[key])
 legacy=json.loads((ROOT/'data/legacy_simulation/manifest.json').read_text());count=0
 for item in legacy['chunks']:
  p=ROOT/'data/legacy_simulation'/item['file']; assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
  z=np.load(p); assert len(z['X'])==len(z['y'])==item['stop']-item['start'];count+=len(z['X'])
 assert count==236575
 print(f'PASS: {len(rows)} saved-prediction runs, disjoint target splits, 2250 compact arcs and {count} legacy arcs.')
if __name__=='__main__':main()
