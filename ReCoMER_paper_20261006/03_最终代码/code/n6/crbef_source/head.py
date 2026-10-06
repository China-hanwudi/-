import torch
class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pa_=torch.nn.Sequential(torch.nn.Linear(1024,256),torch.nn.LayerNorm(256))
            self.pv_=torch.nn.Sequential(torch.nn.Linear(342,256),torch.nn.LayerNorm(256))
            self.mlp=torch.nn.Sequential(torch.nn.Linear(4608,128),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.LayerNorm(128))
            self.out=torch.nn.Linear(128,7);torch.nn.init.zeros_(self.out.bias)
            self.g=torch.nn.Parameter(torch.zeros(1));self.arm='TAV'
        def forward(self,x):
            a=x['a'] if self.arm!='TV' else torch.zeros_like(x['a'])
            v=x['v'] if self.arm!='TA' else torch.zeros_like(x['v'])
            pa=x['pa'] if self.arm!='TV' else torch.zeros_like(x['pa'])
            pv=x['pv'] if self.arm!='TA' else torch.zeros_like(x['pv'])
            z=torch.cat([x['h'],self.pa_(a)*pa[:,None],self.pv_(v)*pv[:,None]],-1)
            return torch.tanh(self.g)*self.out(self.mlp(z))
