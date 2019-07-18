# !/usr/bin/env python
# -*- coding:utf-8 -*-
#author: Shuning Cao

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim.lr_scheduler as LrScheduler
import data_generator as dg
from data_generator import DATASET
from tensorboardX import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1]))

# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='model/DnCNN715_morning', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
# parser.add_argument('--train_data', default='data/Train400', type=str, help='path of train data')###????????????????????????
# parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=2000, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')

args= parser.parse_args()

batch_size = args.batch_size
use_cuda = torch.cuda.is_available()
n_epoch = args.epoch
# sigma = args.sigma
# filename = 'D:\\DATA\\traindata710\\traindata625\\estK'
# filename1 = 'D:\\DATA\\traindata710\\traindata625\\label'
# filename = '/home/solanliu/caoshuning/Kernel_modify/train'# filename1 = '/home/solanliu/caoshuning/Kernel_modify/h.tif'
filename = 'test2/EstimateKernel'
filename1 = 'test2/label'

# save_dir = os.path.join(args.model+'1')
save_dir = os.path.join(args.model)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# 搭建网络
class DnCNN(nn.Module):
    def __init__ (self, depth = 17, n_channels = 64, image_channels = 1, use_bnorm = True, kernel_size = 3):
        super(DnCNN, self). __init__ ()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels = n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append( nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,  bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x) ##########must be data error
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
#
# def mixup_data(x, y, alpha=1.0, use_cuda=True):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#
#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)
#
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam
#
# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class sum_squard_error(_Loss):
    # ...
    # Definition: sum_squard_error = 1/2 * nn.MSELoss(reduction='sum')
    # The backward is defined as:input - target
    # ...
    def __init__ (self, size_average=None, reduce = None, reduction='sum'):
        super(sum_squard_error, self). __init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)). div_2
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)
#     what is input, what is target.  input=DNCNN.out???, target=kt-k???


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

class Polyscheduler(_LRScheduler):
    def __init__(self, optimizer, gamma, EPOCH, last_epoch=-1):
        self.gamma = gamma
        self.EPOCH = EPOCH
        super().__init__(optimizer,last_epoch)


    def get_lr(self):
        return [base_lr * (1. - float((self.last_epoch)/self.EPOCH)) ** self.gamma
        for base_lr in self.base_lrs]


if __name__ == '__main__':
    # model selection
    print('===> Building model')
    model = DnCNN()
    # model = torch.nn.DataParallel(net)

    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()
    # criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
    criterion = sum_squard_error()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = Polyscheduler(optimizer,EPOCH=n_epoch,gamma=0.8)  # learning rates
    # scheduler = LrScheduler.ReduceLROnPlateau(optimizer, 'min')  # learning rates

    writer = SummaryWriter(comment='train')

    DATA = dg.datagenerator(data_dir=filename)
        # data = torch.rand(20, 1, 11, 11)
    DATA = DATA.astype('float32') / 255.0
    DATA = torch.from_numpy(DATA.transpose((0, 3, 1, 2)))
    LABEL = dg.datagenerator(data_dir=filename1)
        # LABEL = torch.rand(20, 1, 11, 11)
    LABEL = LABEL.astype('float32') / 255.0
    LABEL = torch.from_numpy(LABEL.transpose((0, 3, 1, 2)))



    torch_dataset = DATASET(DATA, LABEL)


    for epoch in range(initial_epoch, n_epoch):
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        LR = torch.FloatTensor(scheduler.get_lr())
        DLoader = DataLoader(dataset=torch_dataset, num_workers=4, drop_last=True, batch_size=batch_size, shuffle=True)
        epoch_loss = 0
        start_time = time.time()

        for n_count, batch_yx in enumerate(DLoader):
            optimizer.zero_grad()
            if use_cuda:
                batch_x, batch_y = batch_yx[0].cuda(), batch_yx[1].cuda()
            # inputs, targets_a, targets_b, lam = mixup_data(batch_x, batch_y,
            #                                                       args.alpha, use_cuda)
            loss = criterion(model(batch_x), batch_y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch + 1, n_count, DATA.size(0)//batch_size, loss.item() / batch_size))
        elapsed_time = time.time() - start_time

        log('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch + 1, epoch_loss / (n_count), elapsed_time))
        np.savetxt('train_result.txt', np.hstack((epoch + 1, epoch_loss / n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch + 1)))
        writer.add_scalar('loss_curve',epoch_loss,epoch)
        writer.add_scalar('lr_curve',LR,epoch)
    dummy_input = dummy_input = torch.rand(20, 1, 11, 11)
    writer.add_graph(model, dummy_input)
    writer.close()