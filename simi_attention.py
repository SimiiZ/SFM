from builtins import int
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.transformers.models.gpt2.gpt2_parity import score
# from tensorflow.compiler.tf2xla.python.xla import self_adjoint_eig


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class simi_lbn(nn.Module):
    def __init__(self, cin, ln_ratio=0.5):
        super(simi_lbn, self).__init__()
        print('ln_ratio:', ln_ratio)
        assert ln_ratio <= 1.0 and ln_ratio >= 0.0
        inc = int(cin * ln_ratio)
        self.inc = inc
        self.bnc = cin - self.inc
        if self.inc < 1:
            self.bn = nn.BatchNorm1d(self.bnc, affine=True)
        elif self.inc == cin:
            self.ln = nn.BatchNorm1d(self.inc, affine=True)
        else:
            self.bn = nn.BatchNorm1d(self.bnc, affine=True)
            self.ln = nn.LayerNorm(self.inc, elementwise_affine=True)

    def forward(self, x):

        if self.inc<1:
            return self.bn(x)
        elif self.inc == x.size(1):
            return self.ln(x)
        else:
            bnx = self.bn(x[:, 0:self.bnc])
            lnx = self.ln(x[:, self.bnc:])
            return torch.cat((bnx, lnx), dim=1)

class simi_gep(nn.Module):
    def __init__(self, dims,p=3.0):
        super(simi_gep, self).__init__()
        self.dims = dims
        self.p = p
        self.eps = 1e-6


    def forward(self, x):
        n = x.size(0)
        y = torch.mean(x.clamp(min=self.eps).pow(self.p), dim=self.dims).pow(1.0 / self.p)
        y = y.view(n, -1)
        return y

class simi_attention(nn.Module):
    def __init__(self, indim, r=2.0, ln_ratio=0.5, dims=None):
        super(simi_attention, self).__init__()

        self.attetion = nn.Sequential(
            simi_gep(dims=dims, p=3.0),
            nn.Linear(indim, int(indim // r)),
            simi_lbn(int(indim // r), ln_ratio=ln_ratio),
            nn.ReLU(),
            nn.Linear(int(indim // r), indim),
            simi_lbn(indim, ln_ratio=ln_ratio)
        )
        self.dims = dims


    def forward(self, x):
        return self.attetion(x)

class simi_edge(nn.Module):
    def __init__(self, in_channels, h, w, cmiddim, p_ratio=2.0, ln_ratio=0.5):
        super(simi_edge, self).__init__()

        self.down = nn.Conv2d(in_channels, cmiddim, kernel_size=1, padding=0)

        self.ca = simi_attention(indim=cmiddim, r=p_ratio, ln_ratio=ln_ratio,dims=(2,3))
        self.ha = simi_attention(indim=h, r=p_ratio, ln_ratio=ln_ratio,dims=(1,3))
        self.wa = simi_attention(indim=w, r=p_ratio, ln_ratio=ln_ratio, dims=(1,2))

        self.up = nn.Sequential(
            nn.Conv2d(cmiddim, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.constant_(self.up[1].weight, 0.0)
        nn.init.constant_(self.up[1].bias, 0.0)


    def forward(self, x):

        x = self.down(x)
        n, c, h, w = x.size()
        # print(n,c,h,w)
        ca = self.ca(x).view(n, c, 1, 1)
        ha = self.ha(x).view(n, 1, h, 1)
        wa = self.wa(x).view(n, 1, 1, w)
        z = self.up(ca + ha + wa)

        return z

class simi_area(nn.Module):
    def __init__(self, in_channels, h, w, cmiddim, p_ratio=2.0, ln_ratio=0.5):
        super(simi_area, self).__init__()

        self.down = nn.Conv2d(in_channels, cmiddim, kernel_size=1, padding=0)

        self.hwa = simi_attention(indim=h*w, r=p_ratio**2, ln_ratio=ln_ratio, dims=1)
        self.cwa = simi_attention(indim=cmiddim*w, r=p_ratio**2, ln_ratio=ln_ratio,dims=2)
        self.cha = simi_attention(indim=cmiddim*h, r=p_ratio**2, ln_ratio=ln_ratio,dims=3)

        self.up = nn.Sequential(
            nn.Conv2d(cmiddim, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        nn.init.constant_(self.up[1].weight, 0.0)
        nn.init.constant_(self.up[1].bias, 0.0)


    def forward(self, x):
        x = self.down(x)
        n, c, h, w = x.size()
        hwa = self.hwa(x).view(n, 1, h, w)
        cwa = self.cwa(x).view(n, c, 1, w)
        cha = self.cha(x).view(n, c, h, 1)
        z = self.up(hwa + cwa + cha)


        return z

class simi_comprehensive_atteiton(nn.Module):
    def __init__(self, in_channels, h, w, c_ratio=32.0, p_ratio=2.0, ln_ratio=0.5):
        super(simi_comprehensive_atteiton, self).__init__()
        cmiddim = int(in_channels / c_ratio)

        self.edge = simi_edge(in_channels, h, w, cmiddim, p_ratio=p_ratio, ln_ratio=ln_ratio)
        self.area = simi_area(in_channels, h, w, cmiddim, p_ratio=p_ratio, ln_ratio=ln_ratio)

    def forward(self, x):
        a = self.edge(x)
        b = self.area(x)
        x = x + a + b
        return x



