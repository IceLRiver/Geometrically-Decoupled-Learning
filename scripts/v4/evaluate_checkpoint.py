"""Evaluate a checkpoint on the frozen primary test cohort or the full test partition."""
from pathlib import Path
import argparse,json,time
import numpy as np
from benchmark_data import load_data,freeze_protocol,ROOT
from neural_predict import load_checkpoint,predict,metrics
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['public','simulation','real'],required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--full-test',action='store_true');ap.add_argument('--device',default='cuda');a=ap.parse_args();X,y,_,idx=load_data(a.dataset);ids=idx['test'] if a.full_test else np.array(freeze_protocol()['datasets'][a.dataset]['test_primary']);model,c=load_checkpoint(a.checkpoint,a.device);start=time.perf_counter();state=predict(model,c,X[ids]);result=metrics(state,y[ids]);result.update(dataset=a.dataset,checkpoint=Path(a.checkpoint).name,inference_seconds=time.perf_counter()-start);out=ROOT/'evidence';np.savez_compressed(out/(a.dataset+'_'+Path(a.checkpoint).stem+'_predictions.npz'),indices=ids,position=state[:,:3],velocity=state[:,3:]);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
