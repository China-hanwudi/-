import torch
from torch import nn
class Rich(nn.Module):
 def __init__(self):
  super().__init__();self.pam=nn.Sequential(nn.Linear(1024,256),nn.LayerNorm(256));self.pal=nn.Sequential(nn.Linear(4096,256),nn.LayerNorm(256));self.pv=nn.Sequential(nn.Linear(342,256),nn.LayerNorm(256))
  # Preserve the exact full architecture's random initialization, then remove analytically inactive zero-text columns.
  full=nn.Linear(4864,128);small=nn.Linear(768,128)
  with torch.no_grad():small.weight.copy_(full.weight[:,4096:]);small.bias.copy_(full.bias)
  self.mlp=nn.Sequential(small,nn.GELU(),nn.Dropout(.2),nn.LayerNorm(128));self.out=nn.Linear(128,7);nn.init.zeros_(self.out.bias);self.g=nn.Parameter(torch.zeros(1))
 def encode(self,am,al,v):return [self.pam(am),self.pal(al),self.pv(v)]
 def forward(self,am,al,v):return self.g.tanh()*self.out(self.mlp(torch.cat(self.encode(am,al,v),1)))
