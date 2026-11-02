from paths import BASE
import json,numpy as np
from data_v8 import load_data,protocol
E=BASE/'evidence/v8';primary=json.loads((E/'results.json').read_text());duration=json.loads((E/'duration.json').read_text());count=0

def error(p,y):
 d=p-y.astype(float);return [float(np.sqrt(np.mean(np.sum(d[:,:3]**2,1)))),float(np.sqrt(np.mean(np.sum(d[:,3:]**2,1))))]
for ds in ['public','legacy']:
 X,y,_,_=load_data(ds);ids=np.asarray(protocol()['datasets'][ds]['test_primary'])
 for row in [r for r in primary if r['dataset']==ds]:
  z=np.load(E/'predictions'/f"{ds}__{row['method']}.npz");assert np.array_equal(z['indices'],ids);assert np.allclose(error(z['state'],y[ids]),[row['position_km_rmse'],row['velocity_km_s_rmse']],rtol=1e-10);count+=1
 for m in ['GDL','B-spline-KAN-2025','RGA-KAN-2026','TimeFormer-2026']:
  z=np.load(E/'duration_predictions'/f'{ds}__{m}.npz')
  for k,N in enumerate(z['measurements']):
   row=next(r for r in duration if r['dataset']==ds and r['method']==m and r['measurements']==N);assert np.allclose(error(z['state'][k],y[z['indices']]),[row['position_km_rmse'],row['velocity_km_s_rmse']],rtol=1e-10);count+=1
 print(ds,'primary and duration metrics verified',flush=True)
print('PASS',count,'public/simulation metric rows')

z=np.load(E/'controlled_noise/inputs.npz');checks=0
for row in json.loads((E/'controlled_noise.json').read_text()):
 saved=np.load(E/'controlled_noise'/(row['method']+'_predictions.npz'))['state'];mask=z['additional_noise_arcsec']==row['additional_noise_arcsec'];assert np.allclose(error(saved[mask],z['y'][mask]),[row['position_km_rmse'],row['velocity_km_s_rmse']],rtol=1e-10);checks+=1
print('PASS',checks,'current paired-noise metric rows')
