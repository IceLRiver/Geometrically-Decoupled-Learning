import json,time,copy
import numpy as np
import torch
from data_v5 import ROOT,load_data
from corrected_inference import load_corrected,inputs
from linear_head_utils import fit_head,design
torch.set_num_threads(4)
def main():
 path=ROOT/'evidence/corrected_original/w64_s42_recovered.pt';bundle=load_corrected(path);head=bundle['head'];X,y,_,idx=load_data('legacy');ids=np.r_[idx['train'],idx['val']];tr=np.arange(len(idx['train']));va=np.arange(len(tr),len(ids));F,oldrho,C,L,DC,DL=inputs(bundle,X[ids]);Y=torch.tensor(y[ids],device='cuda',dtype=torch.float32)
 with torch.no_grad():h=head.embed(((F-head.f_mean)/head.f_std).clamp(-12,12));phi=design(head.range,h);rho=(Y[:,:3]-C).norm(dim=1,keepdim=True)
 zero=torch.zeros((1,1),device='cuda',dtype=torch.float64);r_record=fit_head(head.range,phi,rho-oldrho,tr,va,None,zero,head.rho_res_std.double());del phi
 with torch.no_grad():rhat=oldrho+head.rho_res_std*head.range(h);p=C+rhat*L;phi=design(head.velocity,torch.cat([h,p/42000],1));target=Y[:,3:]-DC-rhat*DL
 v_record=fit_head(head.velocity,phi,target,tr,va,None,head.vres_mean.double(),head.vres_std.double());er=ev=0
 with torch.no_grad():
  for j in range(0,len(va),1024):
   k=va[j:j+1024];p,v,_=head(F[k],oldrho[k],C[k],L[k],DC[k],DL[k]);er+=((p-Y[k,:3])**2).sum().item();ev+=((v-Y[k,3:])**2).sum().item()
 er=(er/len(va))**.5;ev=(ev/len(va))**.5;scales=[np.linalg.norm(y[idx['train'],:3].astype(float).std(0)),np.linalg.norm(y[idx['train'],3:].astype(float).std(0))];c=bundle['checkpoint'];c['validation_before_affine_refit']=c['validation'];c['validation']=dict(position_km_rmse=er,velocity_km_s_rmse=ev,common_score=er/scales[0]+ev/scales[1]);c['state']=head.state_dict();c['head_refit']=dict(range=r_record,velocity=v_record);c['config']['affine_refit']=True;dest=path.with_name(path.stem+'_affine.pt');torch.save(c,dest);dest.with_suffix('.json').write_text(json.dumps(dict(validation=c['validation'],head_refit=c['head_refit']),indent=2));print(c['validation'],flush=True)
if __name__=='__main__':main()
