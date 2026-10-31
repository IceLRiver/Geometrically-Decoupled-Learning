"""Independent NumPy checks of checksums, partitions and released prediction metrics."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'scripts/v5'))
from data_v5 import load_data,protocol
def rmse(state,y):
 d=state.astype(float)-y.astype(float);return [float(np.sqrt(np.mean(np.sum(d[:,k:k+3]**2,axis=1)))) for k in [0,3]]
def main():
 manifest=json.loads((BASE/'RELEASE_MANIFEST.json').read_text())
 for item in manifest['files']:
  p=BASE/item['file'];assert p.stat().st_size==item['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],p
 rows=json.loads((BASE/'evidence/v5/results.json').read_text());total=0
 for ds,n in [('public',9632),('simulation',120000),('legacy',236575)]:
  X,y,_,idx=load_data(ds);assert X.shape==(n,40,6) and y.shape==(n,6);assert np.isfinite(X).all() and np.isfinite(y).all();assert np.allclose(np.linalg.norm(X[:,:,3:],axis=-1),1,atol=2e-7);ids=protocol()['datasets'][ds]['test_primary'];assert len(ids)==300 and set(ids)<=set(idx['test']);assert not(set(idx['train'])&set(idx['val']) or set(idx['train'])&set(idx['test']) or set(idx['val'])&set(idx['test']))
  for row in [r for r in rows if r['dataset']==ds]:
   with np.load(BASE/'evidence/v5/predictions'/f"{ds}__{row['method']}.npz") as z:
    assert z['indices'].tolist()==ids;state=z['state'];assert np.isfinite(state).all();a,b=rmse(state,y[z['indices']]);assert np.allclose([a,b],[row['position_km_rmse'],row['velocity_km_s_rmse']],rtol=1e-10,atol=1e-9),(ds,row['method'],a,b);total+=1
  if ds=='legacy':
   full=json.loads((BASE/'evidence/v5/original_fulltest.json').read_text())
   for filename,row in zip(['legacy_fulltest_original.npz','legacy_fulltest_GDL.npz'],full):
    with np.load(BASE/'evidence/v5/predictions'/filename) as z:
     assert np.array_equal(z['indices'],idx['test']);assert len(z['indices'])==47315;a,b=rmse(z['state'],y[z['indices']]);assert np.allclose([a,b],[row['position_km_rmse'],row['velocity_km_s_rmse']],rtol=1e-10)
 z=np.load(BASE/'data/controlled_angle_noise/observations.npz');assert z['X'].shape==(80,40,6);assert np.allclose(np.linalg.norm(z['X'][:,:,3:],axis=-1),1,atol=1e-12);nr=json.loads((BASE/'evidence/v5/controlled_noise.json').read_text())
 for row in nr:
  state=np.load(BASE/'evidence/v5/controlled_noise'/f"{row['method']}_predictions.npz")['state'];mask=z['additional_noise_arcsec']==row['additional_noise_arcsec'];assert mask.sum()==20;a,b=rmse(state[mask],z['y'][mask]);assert np.allclose([a,b],[row['position_km_rmse'],row['velocity_km_s_rmse']],rtol=1e-10)
 print(f'PASS: all file checksums, dataset partitions, {total} public/simulation prediction rows, two full 47,315-arc reproductions and 40 paired-noise result rows.')
if __name__=='__main__':main()
