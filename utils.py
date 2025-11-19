import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
import os
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import torch
from collections import defaultdict
import copy
import random
import math

# coding: utf-8

import numpy as np
import torch
import torch.nn.functional as F


def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class DiyCrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)



def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
    

def GenIdx( train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos
    
def GenCamIdx(gall_img, gall_label, mode):
    if mode =='indoor':
        camIdx = [1,2]
    else:
        camIdx = [1,2,4,5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))
    
    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k,v in enumerate(gall_label) if v==unique_label[i] and gall_cam[k]==camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos
    
def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
            # cam_id = 2
        gall_cam.append(cam_id)
    
    return np.array(gall_cam)
    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        
        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False).astype(int)    
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N          

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
            
def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def set_requires_grad(nets, requires_grad=False):
            """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
            if not isinstance(nets, list):
                nets = [nets]
            for net in nets:
                if net is not None:
                    for param in net.parameters():
                        param.requires_grad = requires_grad


class RandomIdentitySampler_DM(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):
        # print('...........................')
        self.batch_size = batchSize*num_pos
        # self.num_instances = num_pos*2
        self.num_instances_one_modal= num_pos
        self.num_pids_per_batch = batchSize
        # print(self.batch_size,self.num_pids_per_batch,self.num_instances_one_modal)



        self.index_dic_vis = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for i in range(len(color_pos)):
            t=np.array(color_pos[i])
            # print(t.shape)
            self.index_dic_vis[i].append(t)
        # print(self.index_dic_vis)
        self.pids_vis = list(self.index_dic_vis.keys())
        # print('*******',len(self.pids_vis))



        self.index_dic_the = defaultdict(list)
        for i in range(len(thermal_pos)):
            t = np.array(thermal_pos[i])
            self.index_dic_the[i].append(t)
        self.pids_the = list(self.index_dic_the.keys())





        # estimate number of examples in an epoch
        self.length_vis = 0
        for pid in self.pids_vis:
            idxs = self.index_dic_vis[pid]
            num = len(idxs)
            if num < self.num_instances_one_modal:
                num = self.num_instances_one_modal
            self.length_vis += num - num % self.num_instances_one_modal



        self.length_the = 0
        for pid in self.pids_the:
            idxs = self.index_dic_the[pid]
            num = len(idxs)
            if num < self.num_instances_one_modal:
                num = self.num_instances_one_modal
                # print('.....',pid)
            self.length_the += num - num % self.num_instances_one_modal


        self.length = min(self.length_the,self.length_vis)
        # print('........................................',self.length_the,self.length_vis)
        self.pids = list(set(self.pids_vis).intersection(set(self.pids_the)))
        # print(len(self.pids))






    def __iter__(self):

        batch_idxs_dict_vis = defaultdict(list)
        batch_idxs_dict_the = defaultdict(list)

        for pid in self.pids:

            idxs_vis = np.array(copy.deepcopy(self.index_dic_vis[pid])).squeeze()
            # print(np.array(idxs_vis).squeeze())
            # print(idxs_vis[])
            idxs_the = np.array(copy.deepcopy(self.index_dic_the[pid])).squeeze()

            if len(idxs_vis) < self.num_instances_one_modal:
                idxs_vis = np.random.choice(idxs_vis, size=self.num_instances_one_modal, replace=True)

            if len(idxs_the) < self.num_instances_one_modal:
                idxs_the = np.random.choice(idxs_the, size=self.num_instances_one_modal, replace=True)

            random.shuffle(idxs_vis)
            random.shuffle(idxs_the)
            batch_idxs_vis = []
            for idx in idxs_vis:
                batch_idxs_vis.append(idx)
                if len(batch_idxs_vis) == self.num_instances_one_modal:
                    batch_idxs_dict_vis[pid].append(batch_idxs_vis)
                    batch_idxs_vis = []

            batch_idxs_the = []
            for idx in idxs_the:
                batch_idxs_the.append(idx)
                if len(batch_idxs_the) == self.num_instances_one_modal:
                    batch_idxs_dict_the[pid].append(batch_idxs_the)
                    batch_idxs_the = []

        # avai_pids = copy.deepcopy(self.pids)
        # # final_idxs = []
        # vis_final_idxs = []
        # the_final_idxs = []
        # while len(avai_pids) >= self.num_pids_per_batch:
        #     selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
        #     for pid in selected_pids:
        #         batch_idxs_vis = np.array(batch_idxs_dict_vis[pid].pop(0)).reshape(1, -1)
        #         batch_idxs_the = np.array(batch_idxs_dict_the[pid].pop(0)).reshape(1, -1)
        #         # print(batch_idxs_vis)
        #         vis_final_idxs.extend(batch_idxs_vis)
        #         the_final_idxs.extend(batch_idxs_the)
        #         if len(batch_idxs_dict_vis[pid]) == 0 or len(batch_idxs_dict_the[pid]) == 0:
        #             avai_pids.remove(pid)
        #
        # self.index1 = np.array(vis_final_idxs)
        # # print(len(self.index1))
        # self.index2 = the_final_idxs
        #
        # return iter(np.arange(len(self.index1)))

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs_vis = np.array(batch_idxs_dict_vis[pid].pop(0)).reshape(1, -1)
                batch_idxs_the = np.array(batch_idxs_dict_the[pid].pop(0)).reshape(1, -1)
                # print(batch_idxs_vis)
                X = np.concatenate((batch_idxs_vis, batch_idxs_the), axis=0).transpose()
                # print(X)
                final_idxs.extend(X)
                # final_idxs.extend([batch_idxs_vis,batch_idxs_the])
                if len(batch_idxs_dict_vis[pid]) == 0 or len(batch_idxs_dict_the[pid]) == 0:
                    avai_pids.remove(pid)
        # print('xxxxxxxx',len(final_idxs))
        # print(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length