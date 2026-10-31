"""Run an executed literature baseline on the same prespecified test arcs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from data_v5 import ROOT,BASE,load_data,protocol as freeze_protocol
from physics_baselines import solve
from piod_2026_j2 import solve_j2
from alpha_iod_2025 import alpha_iod_cadence
def job(t):
 i,X,method=t;start=time.perf_counter()
 try:
  state,detail=solve_j2(X,42+i) if method=='PIOD-2026-J2' else alpha_iod_cadence(X,42+i) if method=='alpha-IOD-2025' else solve(X,method,42+i);return i,state,dict(details=detail,seconds=time.perf_counter()-start)
 except Exception as e:return i,np.full(6,np.nan),dict(error=repr(e),seconds=time.perf_counter()-start)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['public','simulation','legacy','real'],required=True);ap.add_argument('--method',choices=['Gauss','Laplace','Gooding','PIOD','XGEO-PIOD','alpha-IOD'],required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();X,y,_,_=load_data(a.dataset);ids=freeze_protocol()['datasets'][a.dataset]['test_primary'];pred=[];details=[]
 implementation={'PIOD':'PIOD-2023','XGEO-PIOD':'PIOD-2026','alpha-IOD':'alpha-IOD-2025'}.get(a.method,a.method)
 if implementation=='PIOD-2026':
  selected=Path(__file__).resolve().parents[2]/'evidence/v5/selected_models.json'
  implementation=json.loads(selected.read_text())['datasets'][a.dataset]['methods']['PIOD-2026']['implementation']
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for i,state,d in pool.map(job,[(i,X[i],implementation) for i in ids]):pred.append(state);details.append(dict(index=i,implementation=implementation,**d))
 out=ROOT/'evidence';np.savez_compressed(out/(a.dataset+'_'+a.method+'_predictions.npz'),indices=np.array(ids),position=np.array(pred)[:,:3],velocity=np.array(pred)[:,3:]);(out/(a.dataset+'_'+a.method+'_details.json')).write_text(json.dumps(details,indent=2));print('Saved',len(ids),'arc predictions to',out)
if __name__=='__main__':main()
