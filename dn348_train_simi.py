from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from data_loader import DN348Data, TestDataOld1
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm, eval_rgbn300
from model import embed_net_simi as embed_net
from utils import *
from loss import QuadTripletLoss, Fisher
from tensorboardX import SummaryWriter
from random_erasing import RandomErasing
from loss import pdist_np as eudist

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='DN348', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
# parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
# parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=256, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=256, type=int, metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')


parser.add_argument('--num_pos', default=6, type=int, help='num of pos per identity in each modality')
parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')

parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='v2t', type=str, help='v2t, t2v')
parser.add_argument('--attpos', default=[0,0,0,0], type=int, help='position of attention layer')
parser.add_argument('--triw', default=1.0, type=float, help='weight for triloss')
parser.add_argument('--lnratio', default=0.5, type=float, help='ration for ln')
parser.add_argument('--cratio', default=32.0, type=float, help='ratio for down and up')
parser.add_argument('--pratio', default=2.0, type=float, help='ratio for area and edge')
parser.add_argument('--fisherw', default=1.0, type=float, help='weight for fisher loss')
parser.add_argument('--log_path', default='MAMA1/', type=str, help='log save path')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
set_seed(args.seed)

dataset = args.dataset
assert(dataset=='DN348')
data_path = '/DN/dn348/'
log_path = args.log_path
checkpoint_path = args.log_path
if not os.path.isdir(log_path):
    os.makedirs(log_path)
suffix = dataset
suffix = suffix + '/attpos{}_tri{}_lnr{}_cr{}_pr{}_fisherw{}'.format(args.attpos, args.triw,args.lnratio, args.cratio, args.pratio, args.fisherw)





sys.stdout = Logger(log_path + '/' + suffix + '/log.txt')

vis_log_dir = args.log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability = args.erasing_p, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]), #hlh version

])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w),interpolation=InterpolationMode.BICUBIC,antialias=True),
    transforms.ToTensor(),
    normalize,
])


end = time.time()

trainset = DN348Data(data_path, args.trial, transform=transform_train, img_size=(args.img_w, args.img_h))


# generate the idx of each person identity
color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

# testing set
query_img, query_label, query_datapath_simi = process_test_dn348(data_path, modal=args.mode, doquery=True)
gall_img, gall_label, gall_datapath_simi = process_test_dn348(data_path, modal=args.mode, doquery=False)

gallset = TestDataOld1( gall_datapath_simi, gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestDataOld1( query_datapath_simi, query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building modelbyhlh..')

net = embed_net(n_class, arch=args.arch,lnr=args.lnratio,cr=args.cratio,pr=args.pratio,attpos=args.attpos,imgh=args.img_h,imgw=args.img_w)
net.to(device)
cudnn.benchmark = True




#loss

#loss
criterion_id = nn.CrossEntropyLoss()

loader_batch = args.batch_size * args.num_pos
criterion_tri = QuadTripletLoss()

criterion_seman = Fisher()


criterion_id.to(device)
criterion_tri.to(device)
criterion_seman.to(device)

ignored_params = list(map(id, net.bottleneck.parameters())) \
                 + list(map(id, net.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer = optim.SGD([
    {'params': base_params, 'lr': 0.1 * args.lr, 'weight_decay': 5e-4},
    {'params': net.bottleneck.parameters(), 'lr': args.lr, 'weight_decay': 5e-4},
    {'params': net.classifier.parameters(), 'lr': args.lr, 'weight_decay': 5e-4}],
    momentum=0.9, nesterov=True)




def adjust_learning_rate4(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 5:
        lr = args.lr * (epoch + 1) / 5
    elif epoch >= 5 and epoch < 15:
        lr = args.lr
    elif epoch >= 15 and epoch < 20:
        lr = args.lr * 0.1
    elif epoch >= 20 and epoch < 25:
        lr = args.lr * 0.01
    elif epoch>=25:
        lr = args.lr * 0.001

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):

    current_lr = adjust_learning_rate4(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    seman_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        rgb_label = label1.cuda()
        nir_label = label2.cuda()
        labels = torch.cat((rgb_label, nir_label), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        labels = Variable(labels.long().cuda())
        data_time.update(time.time() - end)


        aid_feat, pool_feat, feat, out0 = net(input1, input2)
        loss_id = criterion_id(out0, labels)
        loss_tri, _ = criterion_tri(pool_feat, labels)


        nhlh = aid_feat.size(0)
        rgbfeat = aid_feat[0:nhlh // 2, :]
        irfeat = aid_feat[nhlh // 2:, :]
        loss_seman = criterion_seman(rgbfeat, irfeat)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item())
        if args.fisherw > 0:
            loss = loss_id + loss_tri * args.triw + loss_seman * args.fisherw
        else:
            loss = loss_id + loss_tri * args.triw
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        seman_loss.update(loss_seman.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 500 == 0:
            print('Epoch: [{}][{}/{}] '
                'Time: {:.3f} ({:.3f}) '
                'lr: {:.3f} '
                'Loss: {:.6f} ({:.6f}) '
                'iLoss: {:.6f} ({:.6f}) '
                'TLoss: {:.6f} ({:.6f}) '
                'Accu: {:.2f} '
                'FLoss: {:.6f} ({:.6f}) '.format(
                epoch, batch_idx, len(trainloader), 
                batch_time.val, batch_time.avg,
                current_lr, 
                train_loss.val, train_loss.avg,
                id_loss.val, id_loss.avg,
                tri_loss.val, tri_loss.avg,
                100. * correct / total, 
                seman_loss.val, seman_loss.avg))

                
    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('fisher_loss', seman_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    if args.mode =='v2t':
        print('v2t evaluation........')
    elif args.mode =='t2v':
        print('t2v evaluation.........')
    else:
        print('wrong testing mode....')
        assert(1==0)


    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input)
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input)
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()

    distmat = -eudist(query_feat, gall_feat)
    distmat_att = -eudist(query_feat_att,gall_feat_att)



    # evaluation
    cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
    cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))


    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 70 - start_epoch):  #300:70

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch,
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch >= 0 and epoch % 5 == 0:
        print('Test Epoch: {}'.format(epoch))
    
        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save modelbyhlh
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + '/' + suffix + '/epoch_best.t')
    
        # save modelbyhlh
        if epoch >=0 and epoch % 10 == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path +'/'+suffix + '/epoch_{}.t'.format(epoch))
    
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))