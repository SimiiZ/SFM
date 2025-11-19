from builtins import range

import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F
import random
from simi_attention import simi_comprehensive_atteiton

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.conv1 = model_v.conv1
        self.bn1 = model_v.bn1
        self.relu = model_v.relu
        self.maxpool = model_v.maxpool



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x




class base_resnet_simi(nn.Module):
    def __init__(self, arch='resnet50',lnr=0.5, cr=32.0, pr=2.0, attpos=[0,4,6,0], imgh=None, imgw=None):
        super(base_resnet_simi, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        self.layer1 = model_base.layer1
        self.layer2 = model_base.layer2
        self.layer3 = model_base.layer3
        self.layer4 = model_base.layer4
        print('imagesize:', imgh, imgw)
        print('attention position:', attpos[0],attpos[1],attpos[2],attpos[3])
        print('lnr, cr, pr', lnr, cr, pr)

        self.simi_l1_num = attpos[0]
        assert self.simi_l1_num <= 3
        self.simi_l1_list = nn.ModuleList(
            [simi_comprehensive_atteiton(in_channels=256, h=imgh//4, w=imgw//4, c_ratio=cr, p_ratio=pr,ln_ratio=lnr) for _ in
             range(self.simi_l1_num)]
        )

        self.simi_l2_num = attpos[1]
        assert self.simi_l2_num <= 4
        self.simi_l2_list = nn.ModuleList(
            [simi_comprehensive_atteiton(in_channels=512, h=imgh//8, w=imgw//8, c_ratio=cr, p_ratio=pr,ln_ratio=lnr) for _ in
             range(self.simi_l2_num)]
        )

        self.simi_l3_num = attpos[2]
        assert self.simi_l3_num <= 6
        self.simi_l3_list = nn.ModuleList(
            [simi_comprehensive_atteiton(in_channels=1024, h=imgh//16, w=imgw//16, c_ratio=cr, p_ratio=pr,ln_ratio=lnr) for _ in
             range(self.simi_l3_num)]
        )

        self.simi_l4_num = attpos[3]
        assert self.simi_l4_num <= 3
        self.simi_l4_list = nn.ModuleList(
            [simi_comprehensive_atteiton(in_channels=2048, h=imgh//16, w=imgw//16, c_ratio=cr, p_ratio=pr,ln_ratio=lnr) for _ in
             range(self.simi_l4_num)]
        )

    def forward(self, x):



        start = len(self.layer1) - self.simi_l1_num
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i>=start:
                x = self.simi_l1_list[i-start](x)


        start = len(self.layer2) - self.simi_l2_num
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i>=start:
                x = self.simi_l2_list[i-start](x)



        start = len(self.layer3) - self.simi_l3_num
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i >= start:
                x = self.simi_l3_list[i-start](x)


        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i < self.simi_l4_num:
                x = self.simi_l4_list[i](x)

        return  x





class embed_net_simi(nn.Module):
    def __init__(self,  class_num,  arch='resnet50',lnr=0.5,cr=32.0,pr=2.0,attpos=[0,4,6,0],imgh=None,imgw=None):
        super(embed_net_simi, self).__init__()

        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet_simi(arch=arch,lnr=lnr,cr=cr,pr=pr,attpos=attpos,imgh=imgh,imgw=imgw)


        pool_dim = 2048
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)



        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)



    def forward(self, x1, x2):

        if self.training:
            x = torch.cat((x1, x2), dim=0)
            x = self.visible_module(x)
        else:
            x = self.visible_module(x1)


        x = self.base_resnet(x)


        #aidx
        x = x.view(x.size(0),x.size(1), -1)
        p = 3.0
        x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
        feat = self.bottleneck(x_pool)


        if self.training:
            return feat, x_pool, feat, self.classifier(feat)
        else:
            return F.normalize(feat,dim=1,p=2), F.normalize(feat,dim=1,p=2) + F.normalize(x_pool,dim=1,p=2)


