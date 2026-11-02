import json,sys,hashlib
import numpy as np
import torch
from data_v7 import ROOT
from train_direct_velocity_heads import CorrectedHeads
from rate_heads import RateHeads
from head_design import design
torch.set_num_threads(4)
from head_design import fit_head
fit=fit_head
c=torch.load(ROOT/'evidence/observer_features.pt',map_location='cpu');n=c['n_train'];F=torch.cat([c['H'],c['f'],c['oldrho']/42000],1).double();OR=c['oldrho'].double();C=c['cam'].double();L=c['los'].double();L/=L.norm(dim=1,keepdim=True);Y=c['y'].double();rho=(Y[:,:3]-C).norm(dim=1,keepdim=True);tr=np.arange(n);va=np.arange(n,len(F))
velocity_ready=True
for path in sorted((ROOT/'evidence/observer_heads').glob('*.pt'))+sorted((ROOT/'evidence/observer_full_heads').glob('*.pt'))+(sorted((ROOT/'evidence/velocity_focused_heads').glob('*.pt')) if velocity_ready else []):
 if path.stem.endswith('_refit'):continue
 if path.with_name(path.stem+'_refit.pt').exists():continue
 ck=torch.load(path,map_location='cpu');width=ck['width'];is_rate=ck['config'].get('range_rate',False);stats={k:ck['state'][k] for k in ['f_mean','f_std','rho_res_std','vres_mean','vres_std']};m=(RateHeads if is_rate else CorrectedHeads)(stats,width).double();m.load_state_dict(ck['state']);m.eval()
 F=torch.cat([c['H'],c['f']]+([c['raw']] if ck['config'].get('full_observations',False) else [])+[c['oldrho']/42000],1).double()
 with torch.no_grad():
  h=torch.cat([m.embed(((F[j:j+512]-m.f_mean)/m.f_std).clamp(-12,12)) for j in range(0,len(F),512)]);phi=torch.cat([design(m.range,h[j:j+512]) for j in range(0,len(F),512)])
  rr=fit(m.range,phi,rho-OR,tr,va,None,torch.zeros((1,1),dtype=torch.float64),m.rho_res_std);del phi
  rhat=OR+m.rho_res_std*m.range(h)
  if is_rate:rhat=torch.nn.functional.softplus(rhat)
  p=C+rhat*L;phi=torch.cat([design(m.velocity,torch.cat([h[j:j+512],p[j:j+512]/42000],1)) for j in range(0,len(h),512)])
  target=((Y[:,3:]-c['dc'].double()-rhat*c['dl'].double())*L).sum(1,keepdim=True) if is_rate else Y[:,3:]
  vr=fit(m.velocity,phi,target,tr,va,None,m.vres_mean,m.vres_std);del phi
  v=m.vres_mean+m.vres_std*m.velocity(torch.cat([h[n:],p[n:]/42000],1))
  if is_rate:v=c['dc'][n:].double()+rhat[n:]*c['dl'][n:].double()+v*L[n:]
  er=(p[n:]-Y[n:,:3]).square().sum(1).mean().sqrt().item();ev=(v-Y[n:,3:]).square().sum(1).mean().sqrt().item();val=dict(position_km_rmse=er,velocity_km_s_rmse=ev,common_score=er/c['state_scales'][0]+ev/c['state_scales'][1])
 ck['state']=m.state_dict();ck['validation']=val;ck['head_refit']=dict(range=rr,velocity=vr);dest=path.with_name(path.stem+'_refit.pt');torch.save(ck,dest);dest.with_suffix('.json').write_text(json.dumps(dict(validation=val,head_refit=ck['head_refit']),indent=2));print(width,val,flush=True)
print('OBSERVER CALIBRATION COMPLETE',flush=True)
