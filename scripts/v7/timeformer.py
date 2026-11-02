"""TimeFormer (ESWA 307, 2026, 131040), six-state regression adaptation.
Retains channel-independent multi-scale CNN embeddings, within/between-patch
MoSA, post-softmax exponential decay and a lower-triangular mask.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SeqBN(nn.Module):
 def __init__(self,d):super().__init__();self.bn=nn.BatchNorm1d(d)
 def forward(self,x):return self.bn(x.transpose(1,2)).transpose(1,2)

class MoSA(nn.Module):
 def __init__(self,d,gamma=.1):
  super().__init__();self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.norm1=SeqBN(d);self.norm2=SeqBN(d);self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d));self.gamma=gamma
 def forward(self,x):
  q,k,v=self.qkv(self.norm1(x)).chunk(3,-1);n=x.shape[1];delay=torch.arange(n,device=x.device);delta=delay[:,None]-delay[None,:];mask=delta>=0
  a=torch.softmax(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1]),-1)
  a=a*torch.exp(-self.gamma*delta.clamp_min(0).to(a.dtype))*mask.to(a.dtype)
  x=x+self.proj(a@v);return x+self.ff(self.norm2(x))

class TimeFormer(nn.Module):
 def __init__(self,width=16,scales=2,gamma=.1):
  super().__init__();self.scales=scales;self.width=width
  self.embed=nn.ModuleList([nn.Conv1d(1,width,3,padding=1) for _ in range(scales)])
  self.inner=nn.ModuleList([MoSA(width,gamma) for _ in range(scales)]);self.outer=nn.ModuleList([MoSA(width,gamma) for _ in range(scales)])
  self.patch=nn.ModuleList([nn.Linear(math.ceil(math.sqrt(math.ceil(40/s)))*width,width) for s in range(1,scales+1)])
  self.reduce=nn.ModuleList([nn.Linear(math.ceil(math.sqrt(math.ceil(40/s)))*width,width) for s in range(1,scales+1)])
  self.out=nn.Linear(6*scales*width,6)
 def forward(self,x):
  b,t,c=x.shape;z=x.transpose(1,2).reshape(b*c,1,t);parts=[]
  for i in range(self.scales):
   a=F.avg_pool1d(z,i+1,stride=i+1,ceil_mode=True);a=self.embed[i](a).transpose(1,2);n=a.shape[1];k=math.ceil(math.sqrt(n));a=F.pad(a,(0,0,0,k*k-n));a=a.reshape(b*c*k,k,self.width);a=self.inner[i](a).reshape(b*c,k,k*self.width);a=self.patch[i](a);a=self.outer[i](a);parts.append(self.reduce[i](a.flatten(1)).reshape(b,c*self.width))
  return self.out(torch.cat(parts,-1))
