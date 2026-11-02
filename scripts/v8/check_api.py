from paths import BASE
import sys,json,numpy as np
sys.path.insert(0,str(BASE/'models'))
from geometric_v8 import predict_states
from native_data import load_native
records=[]
for ds in ['public','legacy']:
 z=load_native(ds);X=z['X60'];assert X.shape[1:]==(60,6);assert np.max(abs(np.linalg.norm(X[z['valid60'],:20,3:],axis=2)-1))<1e-6
 for m in ['GDL','B-spline-KAN-2025','RGA-KAN-2026','TimeFormer-2026']:
  saved=np.load(BASE/'evidence/v8/duration_predictions'/f'{ds}__{m}.npz');ids=saved['indices'][:2]
  for k,N in enumerate(saved['measurements']):
   actual=predict_states(ds,m,X[ids,-N:],int(N),'cpu');delta=np.abs(actual-saved['state'][k,:2]);assert np.max(delta)<2e-5,(ds,m,N,delta.max());records.append(dict(dataset=ds,method=m,N=int(N),max_abs_difference=float(delta.max())))
 print(ds,'CPU APIs and processed prefixes verified',flush=True)
(BASE/'runs/v8/api_verification.json').write_text(json.dumps(records,indent=2));print('PASS',len(records),'CPU model/length combinations')
