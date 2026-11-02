import torch
from torch import nn
from fast_kan import FastPhysKANLayer
class CorrectedHeads(nn.Module):
 def __init__(self,stats,width=64):
  super().__init__()
  for k,v in stats.items():self.register_buffer(k,torch.as_tensor(v,dtype=torch.float32))
  self.embed=FastPhysKANLayer(len(stats['f_mean']),width,32);self.range=FastPhysKANLayer(width,1,32);self.velocity=FastPhysKANLayer(width+3,3,32)
 def forward(self,f,oldrho,cam,los,dc,dl):
  h=self.embed(((f-self.f_mean)/self.f_std).clamp(-12,12));rho=oldrho+self.rho_res_std*self.range(h);p=cam+rho*los;v=self.vres_mean+self.vres_std*self.velocity(torch.cat([h,p/42000],1));return p,v,rho
