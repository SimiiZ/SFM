import scipy.io
import torch
import numpy as np
#import time
import os
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import scipy.io
import argparse

######################################################################

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='llcm', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
# parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
# parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')


parser.add_argument('--num_pos', default=30, type=int, help='num of pos per identity in each modality')
parser.add_argument('--batch-size', default=20, type=int, metavar='B', help='training batch size')

parser.add_argument('--gpu', default='6', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='v2t', type=str, help='v2t, t2v')
parser.add_argument('--lkvreuse', default=1, type=float, help=' 1 v_l, k_l reuse, -1 not reuse')
parser.add_argument('--mmdw', default=0.1, type=float, help='weight for mmd')
parser.add_argument('--triw', default=4, type=float, help='weight for triloss')
parser.add_argument('--log_path', default='AGW_vehicle/temp0812/', type=str, help='log save path')

parser.add_argument('--model_path', default='AGW_vehicle/temp0812/', type=str, help='log save path')
parser.add_argument('--save_path', default='./', type=str, help='log save path')
parser.add_argument('--mat_name', default='tsne_mmdw0.1.mat', type=str, help='log save path')

args = parser.parse_args()

mat_name = args.mat_name
result = scipy.io.loadmat(mat_name)
query_feature = torch.FloatTensor(result['query_f'])
query_label = torch.FloatTensor(result['query_label'][0])
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = torch.FloatTensor(result['gallery_label'][0])

query_feature = query_feature.detach().cpu().numpy()
gallery_feature = gallery_feature.detach().cpu().numpy()


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    print(emb1.shape, emb2.shape)
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx

# 计算距离矩阵  
# gallery_feature = np.array(gallery_feature)  # Your NumPy array  
# query_feature = np.array(query_feature)     # Your NumPy array  
gallery_feature = torch.tensor(gallery_feature)  # 或者使用 torch.from_numpy(gallery_feature)  
query_feature = torch.tensor(query_feature)      # 或者使用 torch.from_numpy(query_feature)

distmat = pdist_torch(gallery_feature, query_feature).cuda()  # miv 
# distmat = pdist_torch(gallery_feature, gallery_feature).cuda()  # mii  
# distmat = pdist_torch(query_feature, gallery_feature).cuda()  # mvi
# distmat = pdist_torch(query_feature, query_feature).cuda()  # mvv

# distmat = torch.FloatTensor(1 - np.matmul(query_feature, np.transpose(gallery_feature))).cuda() #Cosine distance


mask = query_label.expand(len(gallery_label), len(query_label)).eq(gallery_label.expand(len(query_label), len(gallery_label)).t()).cuda()  
# mask = gallery_label.expand(len(gallery_label), len(gallery_label)).eq(gallery_label.expand(len(gallery_label), len(gallery_label)).t()).cuda()
# mask = gallery_label.expand(len(query_label), len(gallery_label)).eq(query_label.expand(len(gallery_label), len(query_label)).t()).cuda()
# mask = query_label.expand(len(query_label), len(query_label)).eq(query_label.expand(len(query_label), len(query_label)).t()).cuda()

# 提取 intra-class 和 inter-class 距离  
intra = distmat[mask]  
inter = distmat[mask == 0]  

# 绘制直方图  
plt.rcParams.update({'font.size': 14})  

fig, ax = plt.subplots()  
b = np.linspace(0.5, np.max(distmat.detach().cpu().numpy()), num=1000)  # 更新为距离的范围  

ax.hist(intra.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color='blue', density=True, label='Intra-class')  
ax.hist(inter.detach().cpu().numpy(), b, histtype="stepfilled", alpha=0.6, color='green', density=True, label='Inter-class')  

ax.set_xlabel('Feature Distance')  
ax.set_ylabel('Frequency')  
ax.legend()  

name = args.save_path  
fig.savefig(f'{name}.png', dpi=1000, format='png')  
plt.show()