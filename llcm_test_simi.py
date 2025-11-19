from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from data_loader import SYSUData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net_simi as embed_net
from utils import *
import pdb
from loss import pdist_np as eudist

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='llcm', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
# parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
# parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--num_pos', default=5, type=int, help='num of pos per identity in each modality')
parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')



# parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')


parser.add_argument('--gpu', default='6', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='t2v', type=str, help='v2t, t2v')
parser.add_argument('--attpos', default=[0,4,6,0], type=int, help='position of attention layer')
parser.add_argument('--triw', default=1.0, type=float, help='weight for triloss')
parser.add_argument('--lnratio', default=0.5, type=float, help='ration for ln')
parser.add_argument('--cratio', default=16.0, type=float, help='ratio for down and up')
parser.add_argument('--pratio', default=2.0, type=float, help='ratio for area and edge')
parser.add_argument('--fisherw', default=1.0, type=float, help='weight for fisher loss')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--resume', '-r', default='best', type=str, help='resume from checkpoint')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


dataset = args.dataset
suffix = dataset
suffix = suffix + '/attpos{}_tri{}_lnr{}_cr{}_pr{}_fisherw{}'.format(args.attpos, args.triw,args.lnratio,args.cratio,args.pratio,args.fisherw)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
pool_dim = 2048
print('==> Building model..')



net = embed_net(713, arch=args.arch,lnr=args.lnratio,cr=args.cratio,pr=args.pratio,attpos=args.attpos,imgh=args.img_h,imgw=args.img_w)
net.to(device)
cudnn.benchmark = True




data_path = '/datasets/LLCM/'
print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w),interpolation=InterpolationMode.BICUBIC,antialias=True),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input)
            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc


def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input)
            query_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc



sys.stdout = Logger(args.log_path + '/' + suffix + '/result_'+args.mode+'.txt')

checkpoint_path = args.log_path
if len(args.resume) > 0:
    model_path = checkpoint_path +'/'+suffix+'/epoch_'+ args.resume+'.t'
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loading checkpoint {}'.format(model_path))
    else:
        print('==> wrong... loading checkpoint {}'.format(model_path))
        assert (1 == 0)

    # testing set
query_img, query_label, query_cam = process_query_llcm(data_path, mode=args.mode) # control testing mode
gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=args.mode, trial=0)
nquery = len(query_label)
ngall = len(gall_label)
print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")

queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
for trial in range(10):
    if args.mode =='v2t':
        print('trial_' + str(trial)+ ': v2t evaluation.....')
    elif args.mode =='t2v':
        print('trial_' + str(trial) + ': t2v evaluation.....')
    else:
        print('wrong testing mode....')
        assert(1==0)

    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=args.mode, trial=trial)# control testing mode

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

    # pool5 feature
    distmat_pool = -eudist(query_feat_pool, gall_feat_pool)
    cmc_pool, mAP_pool, mINP_pool = eval_llcm(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

    # fc feature
    distmat = -eudist(query_feat_fc, gall_feat_fc)
    cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)



    if trial == 0:
        all_cmc = cmc
        all_mAP = mAP
        all_mINP = mINP
        all_cmc_pool = cmc_pool
        all_mAP_pool = mAP_pool
        all_mINP_pool = mINP_pool
    else:
        all_cmc = all_cmc + cmc
        all_mAP = all_mAP + mAP
        all_mINP = all_mINP + mINP
        all_cmc_pool = all_cmc_pool + cmc_pool
        all_mAP_pool = all_mAP_pool + mAP_pool
        all_mINP_pool = all_mINP_pool + mINP_pool

    print('Test Trial: {}'.format(trial))
    print(
        'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print(
        'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

cmc = all_cmc / 10
mAP = all_mAP / 10
mINP = all_mINP / 10
cmc_pool = all_cmc_pool / 10
mAP_pool = all_mAP_pool / 10
mINP_pool = all_mINP_pool / 10

print('All Average:')
print(
    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print(
    'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

