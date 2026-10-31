"""Task-regression implementations of ICLR-2025 KAN and CMAME-2026 RGA-KAN.

KAN edge functions: SiLU residual plus cubic B-splines (Liu et al.).
RGA-KAN: source sine embedding, Chebyshev gates and two-layer adaptive blocks
(Rigas et al., doi:10.1016/j.cma.2026.118761). This is a PyTorch port of the
published architecture, not a reproduction of its PDE training experiments.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class SplineLayer(nn.Module):
 def __init__(self,ni,no,grid=5,order=3):
  super().__init__();self.ni=ni;self.no=no;self.order=order
  self.register_buffer('knots',torch.arange(-order,grid+order+1,dtype=torch.float32)*6/grid-3)
  self.base=nn.Parameter(torch.empty(no,ni));self.coef=nn.Parameter(torch.empty(no,ni,grid+order));nn.init.kaiming_uniform_(self.base,a=math.sqrt(5));nn.init.normal_(self.coef,std=.1/(grid*math.sqrt(ni)))
 def basis(self,x):
  q=x.unsqueeze(-1);t=self.knots;b=((q>=t[:-1])&(q<t[1:])).to(x.dtype)
  for k in range(1,self.order+1):b=(q-t[:-(k+1)])/(t[k:-1]-t[:-(k+1)])*b[:,:,:-1]+(t[k+1:]-q)/(t[k+1:]-t[1:-k])*b[:,:,1:]
  return b
 def forward(self,x):
  b=self.basis(x);return F.linear(F.silu(x),self.base)+F.linear(b.flatten(1),self.coef.flatten(1))

class BSplineKAN(nn.Module):
 def __init__(self,ni,no,width=64,grid=5):
  super().__init__();self.layers=nn.ModuleList([SplineLayer(ni,width,grid),SplineLayer(width,no,grid)])
 def forward(self,x):
  for layer in self.layers:x=layer(x)
  return x

def chebyshev(x,D):
 x=torch.tanh(x);a=torch.ones_like(x);b=x;values=[b]
 for i in range(2,D+1):a,b=b,2*x*b-a;values.append(b)
 return torch.stack(values,-1)

class ChebLayer(nn.Module):
 def __init__(self,ni,no,D=5):
  super().__init__();self.ni=ni;self.D=D;sample=torch.linspace(-1,1,10000,requires_grad=True);b=chebyshev(sample[:,None],D)[:,0]
  mu0=(b.detach()**2).mean(0);mu1=torch.stack([(torch.autograd.grad(b[:,i].sum(),sample,retain_graph=True)[0].detach()**2).mean() for i in range(D)])
  sigma=sample.detach().std(unbiased=False)*torch.sqrt(1/(D*(ni*mu0+no*mu1)))
  self.coef=nn.Parameter(torch.randn(no,ni,D)*sigma);self.bias=nn.Parameter(torch.zeros(no))
 def forward(self,x):return F.linear(chebyshev(x,self.D).flatten(1),self.coef.flatten(1),self.bias)

class SineEmbedding(nn.Module):
 def __init__(self,ni,no,D=5,glorot=False):
  super().__init__();self.omega=nn.Parameter(torch.randn(D));self.phase=nn.Parameter(torch.zeros(D));self.coef=nn.Parameter(torch.randn(no,ni,D)/math.sqrt(ni*D));self.bias=nn.Parameter(torch.zeros(no));self.D=D
  if glorot:
   with torch.no_grad():
    sample=torch.linspace(-1,1,10000)[:,None];w=self.omega[None];sd=torch.sqrt(.5*(1-torch.exp(-2*w*w))).clamp_min(1e-8);b=torch.sin(sample*w)/sd;der=w*torch.cos(sample*w)/sd;mu0=(b*b).mean(0);mu1=(der*der).mean(0);sigma=sample.std(unbiased=False)*torch.sqrt(2/(ni*mu0+no*mu1));self.coef.copy_(torch.randn_like(self.coef)*sigma)
 def forward(self,x):
  mu=torch.exp(-.5*self.omega**2)*torch.sin(self.phase);std=torch.sqrt((.5*(1-torch.exp(-2*self.omega**2)*torch.cos(2*self.phase))-mu**2).clamp_min(1e-12));b=(torch.sin(x.unsqueeze(-1)*self.omega+self.phase)+mu)/(std+1e-8)
  return F.linear(b.flatten(1),self.coef.flatten(1),self.bias)

class RGABlock(nn.Module):
 def __init__(self,width,D):
  super().__init__();self.first=ChebLayer(width,width,D);self.second=ChebLayer(width,width,D);self.alpha=nn.Parameter(torch.tensor(0.));self.beta=nn.Parameter(torch.tensor(0.))
 def forward(self,x,u,v):
  identity=x;q=self.first(x);q=q*u+(1-q)*v;q=self.beta*q+(1-self.beta)*identity;q=self.second(q);q=q*u+(1-q)*v
  return self.alpha*q+(1-self.alpha)*identity

class RGAKAN(nn.Module):
 def __init__(self,ni,no,width=64,D=5,blocks=2,glorot_sine=False):
  super().__init__();self.embed=SineEmbedding(ni,width,D,glorot_sine);self.U=ChebLayer(width,width,D);self.V=ChebLayer(width,width,D);self.blocks=nn.ModuleList([RGABlock(width,D) for _ in range(blocks)]);self.output=ChebLayer(width,no,D)
 def forward(self,x):
  x=self.embed(x);u=self.U(x);v=self.V(x)
  for block in self.blocks:x=block(x,u,v)
  return self.output(x)
