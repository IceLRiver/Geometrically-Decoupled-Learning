"""Geometric velocity candidate: v = Rdot + rho Ldot + rhodot L."""
import torch
from torch import nn
from fast_kan import FastPhysKANLayer
class RateHeads(nn.Module):
 def __init__(self,stats,width=64):
  super().__init__()
  for k,v in stats.items():self.register_buffer(k,torch.as_tensor(v,dtype=torch.float32))
  self.embed=FastPhysKANLayer(len(stats['f_mean']),width,32);self.range=FastPhysKANLayer(width,1,32);self.velocity=FastPhysKANLayer(width+3,1,32)
 def forward(self,f,oldrho,cam,los,dc,dl):
  h=self.embed(((f-self.f_mean)/self.f_std).clamp(-12,12));rho=torch.nn.functional.softplus(oldrho+self.rho_res_std*self.range(h));p=cam+rho*los;rate=self.vres_mean+self.vres_std*self.velocity(torch.cat([h,p/42000],1));v=dc+rho*dl+rate*los;return p,v,rho
