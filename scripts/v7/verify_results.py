from pathlib import Path
import sys,json,argparse
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'scripts/v7'))
from data_v7 import load_data,protocol
E=R/'evidence/v7';rows=json.loads((E/'results.json').read_text(encoding='utf-8'));arcs=json.loads((E/'arc_length.json').read_text(encoding='utf-8'));noise=json.loads((E/'controlled_noise.json').read_text(encoding='utf-8'))
def errors(p,y):
 d=np.asarray(p)-np.asarray(y,dtype=float);return np.sqrt(np.mean(np.sum(d[:,:3]**2,1))),np.sqrt(np.mean(np.sum(d[:,3:]**2,1)))
checks=0
physical=[json.loads(line) for line in (E/'arc_length/physics_native.jsonl').read_text(encoding='utf-8').splitlines()]
for ds in ['public','legacy']:
 X,y,_,idx=load_data(ds);ids=np.asarray(protocol()['datasets'][ds]['test_primary'])
 for row in [r for r in rows if r['dataset']==ds and r['category']=='primary']:
  z=np.load(E/'predictions'/(ds+'__'+row['method']+'.npz'));assert np.array_equal(z['indices'],ids);er,ev=errors(z['state'],y[ids]);assert np.isclose(er,row['position_km_rmse'],rtol=1e-10);assert np.isclose(ev,row['velocity_km_s_rmse'],rtol=1e-10);checks+=1
 print(ds,'primary metrics reproduced',flush=True)
 aids=np.load(E/'arc_length'/(ds+'_indices.npz'))['indices']
 for method in dict.fromkeys(r['method'] for r in arcs):
  if method in ['Gauss','Laplace','Gooding','PIOD-2023','alpha-IOD-2025','PIOD-2026']:
   state=np.empty((4,20,6));filled=np.zeros((4,20),bool)
   for r in physical:
    if r['dataset']==ds and r['method']==method:state[r['level'],r['row']]=r['state'];filled[r['level'],r['row']]=True
   assert filled.all()
  else:state=np.load(E/'arc_length'/(ds+'__'+method+'.npz'))['state']
  for k,t in enumerate([9,19,29,39]):
   row=next(r for r in arcs if r['dataset']==ds and r['method']==method and r['elapsed_s']==t);er,ev=errors(state[k],y[aids]);assert np.allclose([er,ev],[row['position_km_rmse'],row['velocity_km_s_rmse']],rtol=1e-10);checks+=1
print('PASS:',checks,'current public/simulation primary rows reproduce from saved predictions.')
