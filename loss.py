import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.debugging import torchscript
from torch.autograd.function import Function
from torch.autograd import Variable
from torch.nn import init


class Fisher(nn.Module):
    def __init__(self, ):
        super(Fisher, self).__init__()


    def forward(self, rgb, ir):


        rgb_mean = rgb.mean(dim=0,keepdim=True)
        ir_mean = ir.mean(dim=0,keepdim=True)

        all = torch.cat([rgb_mean,ir_mean],dim=0)
        all_mean = all.mean(dim=0,keepdim=True)

        inter_calss_var = (rgb_mean-all_mean).pow(2).sum(dim=1).clamp(min=1e-12).sqrt() + (ir_mean-all_mean).pow(2).sum(dim=1).clamp(min=1e-12).sqrt()


        loss = inter_calss_var


        return loss



def sparsemax(x):
    dim = -1
    x = x-x.mean(dim,keepdim=True)
    sorted_x, _ = torch.sort(x, descending=True, dim=dim)
    cumsum_x = sorted_x.cumsum(dim)-1
    k = torch.arange(1,x.size(dim)+1,dtype=x.dtype,device=x.device)
    k = k.expand_as(cumsum_x)
    valid = sorted_x-cumsum_x/k>0
    k_z = valid.sum(dim=dim,keepdim=True)
    tau_z = cumsum_x.gather(dim,k_z-1)/k_z
    output = torch.clamp(x-tau_z,min=0)
    return output

def log_sparsemax(x):
    dim = -1
    x = x-x.mean(dim,keepdim=True)
    sorted_x, _ = torch.sort(x, descending=True, dim=dim)
    cumsum_x = sorted_x.cumsum(dim)-1
    k = torch.arange(1,x.size(dim)+1,dtype=x.dtype,device=x.device)
    k = k.expand_as(cumsum_x)
    valid = sorted_x-cumsum_x/k>0
    k_z = valid.sum(dim=dim,keepdim=True)
    tau_z = cumsum_x.gather(dim,k_z-1)/k_z
    output = torch.clamp(x-tau_z,min=0)
    output = torch.log(output)
    return output

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.01)
        if m.bias:
            init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)




class QuadTripletLoss(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(QuadTripletLoss, self).__init__()
        self.loss1 = nn.SoftMarginLoss()
        self.loss2 = nn.SoftMarginLoss()
        self.loss3 = nn.SoftMarginLoss()
        self.loss4 = nn.SoftMarginLoss()



    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)

        half = inputs.size(0)//2
        input1=inputs[0:half,:]
        input2=inputs[half:, :]
        target1=targets[0:half]
        target2=targets[half:]#nx1
        assert sum(target2!=target1)==0




        dist_mat11 = pdist_torch(input1, input1)
        dist_mat12 = pdist_torch(input1, input2)
        dist_mat21 = pdist_torch(input2, input1)
        dist_mat22 = pdist_torch(input2, input2)

        n = input1.size(0)
        mask1 = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap1, dist_an1 = [], []
        for i in range(n):
            dist_ap1.append(dist_mat11[i][mask1[i]].max().unsqueeze(0))
            dist_an1.append(dist_mat11[i][mask1[i] == 0].min().unsqueeze(0))
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)
        y1 = torch.ones_like(dist_an1)
        loss1 = self.loss1(dist_an1 - dist_ap1, y1)
        correct1 = torch.ge(dist_an1, dist_ap1).sum().item()


        n = input1.size(0)
        mask2 = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap2, dist_an2 = [], []
        for i in range(n):
            dist_ap2.append(dist_mat12[i][mask2[i]].max().unsqueeze(0))
            dist_an2.append(dist_mat12[i][mask2[i] == 0].min().unsqueeze(0))
        dist_ap2 = torch.cat(dist_ap2)
        dist_an2 = torch.cat(dist_an2)
        y2 = torch.ones_like(dist_an2)
        loss2 = self.loss2(dist_an2 - dist_ap2, y2)
        correct2 = torch.ge(dist_an2, dist_ap2).sum().item()


        n = input1.size(0)
        mask3 = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap3, dist_an3 = [], []
        for i in range(n):
            dist_ap3.append(dist_mat21[i][mask3[i]].max().unsqueeze(0))
            dist_an3.append(dist_mat21[i][mask3[i] == 0].min().unsqueeze(0))
        dist_ap3 = torch.cat(dist_ap3)
        dist_an3 = torch.cat(dist_an3)
        y3 = torch.ones_like(dist_an3)
        loss3 = self.loss3(dist_an3 - dist_ap3, y3)
        correct3 = torch.ge(dist_an3, dist_ap3).sum().item()



        n = input1.size(0)
        mask4 = target1.expand(n, n).eq(target1.expand(n, n).t())
        dist_ap4, dist_an4 = [], []
        for i in range(n):
            dist_ap4.append(dist_mat22[i][mask4[i]].max().unsqueeze(0))
            dist_an4.append(dist_mat22[i][mask4[i] == 0].min().unsqueeze(0))
        dist_ap4 = torch.cat(dist_ap4)
        dist_an4 = torch.cat(dist_an4)
        y4 = torch.ones_like(dist_an4)
        loss4 = self.loss4(dist_an4 - dist_ap4, y4)
        correct4 = torch.ge(dist_an4, dist_ap4).sum().item()


        loss = (loss1+loss2+loss3+loss4)*0.25
        correct = (correct1+correct2+correct3+correct4)*0.25


        #std loss
        # n = input1.size(0)
        # # idx = torch.arange(n)
        # # diag = idx.expand(n, n).eq(idx.expand(n, n).t())
        # all_dist = torch.cat((dist_mat11.unsqueeze(2),dist_mat12.unsqueeze(2),dist_mat21.unsqueeze(2),dist_mat22.unsqueeze(2)),dim=2)
        # # std = all_dist.std(dim=2)[~diag].mean()
        # std = all_dist.std(dim=2).mean()



        return loss, correct

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class CenterTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.softranking_loss = nn.SoftMarginLoss()

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # print(mask.size())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        loss = self.softranking_loss(dist_an - dist_ap, y)  # self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        # print(num_classes)
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # print(targets.size())
        # print(targets)
        # if torch.any(torch.isnan(inputs)):
            # print('fuck you')
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class IntraEnhanceLoss(nn.Module):
    def __init__(self,margin=0.0):
        super(IntraEnhanceLoss, self).__init__()

        # self.rgbmiu = torch.zeros(2048,1).cuda()
        # self.irmiu = torch.zeros(2048,1).cuda()

        self.register_buffer("rgbmiu",torch.zeros(2048,1).cuda())
        self.register_buffer("irmiu",torch.zeros(2048,1).cuda())
        self.eta = 0.9




    def forward(self, rgbfeat,irfeat,target=None):
        #v1
        # n = rgbfeat.size(0)
        # dist = pdist_torch(rgbfeat,irfeat)
        # idnum = torch.unique(target).size(0)
        # imageperid =  n//idnum
        # std = 0
        # for i in range(idnum):
        #     posdist = dist[i*imageperid:(i+1)*imageperid,i*imageperid:(i+1)*imageperid]
        #     std = std + posdist.std()
        # std = std/idnum


        #v2
        n = rgbfeat.size(0)
        inter_dist = pdist_torch(rgbfeat,irfeat)
        intra_dist_rgb = pdist_torch(rgbfeat,rgbfeat)
        intra_dist_ir = pdist_torch(irfeat, irfeat)
        idnum = torch.unique(target).size(0)
        imageperid =  n//idnum
        std = 0
        a = torch.arange(imageperid)
        posmask = a.expand(imageperid, imageperid).eq(a.expand(imageperid,imageperid).t())
        negmask = ~posmask





        for i in range(idnum):
            inter_posdist = inter_dist[i*imageperid:(i+1)*imageperid,i*imageperid:(i+1)*imageperid]
            inter_miu = inter_posdist.mean()

            intra_posdist_rgb = intra_dist_rgb[i*imageperid:(i+1)*imageperid,i*imageperid:(i+1)*imageperid][negmask]
            intra_miu_rgb = intra_posdist_rgb.mean()

            intra_posdist_ir= intra_dist_ir[i*imageperid:(i+1)*imageperid,i*imageperid:(i+1)*imageperid][negmask]
            intra_miu_ir = intra_posdist_ir.mean()


            # print(inter_std,intra_std_rgb,intra_std_ir)
            if intra_miu_rgb > intra_miu_ir:
                std = std + F.relu(inter_miu - intra_miu_rgb)
            else:
                std = std + F.relu(inter_miu - intra_miu_ir)

        std = std/idnum





#v5
        # n = rgbfeat.size(0)
        # idnum = torch.unique(target).size(0)
        # imageperid = n // idnum
        #
        # rgbfeat = rgbfeat.view(idnum,imageperid,-1).mean(1).t()
        # irfeat = irfeat.view(idnum,imageperid,-1).mean(1).t() #cxn
        #
        # rgbmiu = torch.mean(rgbfeat,dim=1,keepdim=True)#cx1
        # irmiu = torch.mean(irfeat,dim=1,keepdim=True)#cx1
        # rgbstd = torch.std(rgbfeat,dim=1,keepdim=True)
        # irstd = torch.std(irfeat,dim=1,keepdim=True)
        #
        # cov = (rgbfeat-rgbmiu)*(irfeat-irmiu)#cxn
        # pcc = cov.mean(1,keepdim=True)/(rgbstd*irstd+1e-6) #cx1
        # # print(pcc)
        #
        # z = F.softplus(-pcc).mean(0)



#v4
        # n = rgbfeat.size(0)
        # cur_rgbmiu= rgbfeat.t().mean(1,keepdim=True) #cx1
        # cur_irmiu = irfeat.t().mean(1,keepdim=True) #cx1
        # # print(self.rgbmiu.size(),cur_rgbmiu.size())
        #
        # rgbmiu = self.eta * self.rgbmiu + (1.0 - self.eta) * cur_rgbmiu
        # irmiu = self.eta * self.irmiu + (1.0 - self.eta) * cur_irmiu
        # self.rgbmiu.copy_(rgbmiu.detach())
        # self.irmiu.copy_(irmiu.detach())
        #
        #
        # diff = rgbmiu - irmiu #cxn
        # diff = diff*diff
        # z = diff.sum().sqrt()




        # print(pcc)
        # z = F.softplus(-pcc).mean(0)

        return   std


class myMSE(nn.Module):
    def __init__(self,margin=0.0):
        super(myMSE, self).__init__()
        self.loss = MMDLoss(kernel_type='rbf',kernel_mul=2.0,kernel_num=5,margin=margin)

    def forward(self, rgbfeat,irfeat):
        #v1

        # rgbfeat = rgbfeat.mean(0)
        # irfeat = irfeat.mean(0)
        # diff = rgbfeat - irfeat
        # diff = diff*diff
        # # z = diff.sum(1).sqrt().mean()
        # z = diff.sum().sqrt()
        z = self.loss(rgbfeat,irfeat)

        return z

