from pathlib import Path
import sys,argparse,json
import numpy as np
import torch
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'scripts/v7'))
from data_v7 import load_data,protocol
from inference_v7 import selected,predict
ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['public','legacy'],required=True);ap.add_argument('--method',choices=['GDL','TimeFormer-2026'],default='GDL');ap.add_argument('--device',default='cpu');a=ap.parse_args();torch.set_num_threads(4)
X,y,_,_=load_data(a.dataset);ids=np.asarray(protocol()['datasets'][a.dataset]['test_primary']);p=predict(selected(a.dataset,a.method,a.device),X[ids]);d=p-y[ids].astype(float);print(json.dumps(dict(dataset=a.dataset,method=a.method,arcs=len(ids),position_km_rmse=float(np.sqrt(np.mean(np.sum(d[:,:3]**2,1)))),velocity_km_s_rmse=float(np.sqrt(np.mean(np.sum(d[:,3:]**2,1))))),indent=2))
