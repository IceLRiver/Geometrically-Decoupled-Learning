"""Algebraically identical shared residual evaluation with early input summation."""
import torch
from original_models import PhysKANLayer
class FastPhysKANLayer(PhysKANLayer):
 def forward(self,x):
  phys=x@self.w_x.T+torch.sin(x*self.omega)@self.w_sin.T+torch.cos(x*self.omega)@self.w_cos.T
  hidden=self.res_mlp[1](self.res_mlp[0](x.unsqueeze(-1)))
  if self.use_residual_gate:
   gate=torch.tanh(self.res_gate);hidden=(hidden*gate[None,:,None]).sum(1);mass=gate.sum()
  else:hidden=hidden.sum(1);mass=self.in_dim
  res=hidden@self.res_mlp[2].weight.T
  if self.res_mlp[2].bias is not None:res=res+mass*self.res_mlp[2].bias
  out=phys+res
  return out+self.bias if self.bias is not None else out
