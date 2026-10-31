from pathlib import Path
import sys,numpy as np,torch
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'scripts'));sys.path.insert(0,str(BASE/'scripts/v4'))
import original_models as mod
from fast_kan import FastPhysKANLayer
mod.PhysKANLayer=FastPhysKANLayer
def forward(self,x):
 b=len(x);x3=x.view(b,40,6);e=self.expand_features(x3)
 c=torch.cat([e[:,:1].expand(-1,38,-1),e[:,1:39],e[:,39:].expand(-1,38,-1)],-1)
 h=self.embed(c.reshape(-1,126)).reshape(b,38,-1)
 if self.mid_layers is not None:
  flat=h.reshape(-1,self.d_model)
  for layer in self.mid_layers:flat=flat+layer(flat)
  h=h+flat.reshape_as(h)
 h=self.drop(self.norm(h))
 if self.temporal_pool=='mean':h=h.mean(1)
 elif self.temporal_pool=='max':h=h.max(1).values
 else:h=(h*torch.softmax(self.attn_score(h),1)).sum(1)
 rho=self.f1(h)*self.rho_std+self.rho_mean;cam=x3[:,-1,:3]*self.cam_std+self.cam_mean;los=x3[:,-1,3:]*self.los_std+self.los_mean;los=los/(los.norm(dim=1,keepdim=True)+1e-8);pos=cam+rho*los;vel=self.f2(torch.cat([h,pos],-1))
 return dict(xyz=pos,vxyz=vel,rho=rho)
mod.OrbitPhysKAN.forward=forward

def encode(model,x):
 b=len(x);e=model.expand_features(x.view(b,40,6));c=torch.cat([e[:,:1].expand(-1,38,-1),e[:,1:39],e[:,39:].expand(-1,38,-1)],-1);h=model.embed(c.reshape(-1,126)).reshape(b,38,-1);flat=h.reshape(-1,model.d_model)
 for layer in model.mid_layers:flat=flat+layer(flat)
 h=model.norm(h+flat.reshape_as(h));return (h*torch.softmax(model.attn_score(h),1)).sum(1)

