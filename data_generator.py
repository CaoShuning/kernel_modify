# !/usr/bin/env python
# -*- coding:utf-8 -*-
#author: Shuning Cao

import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size =4
# filename = 'D:\\DATA\\traindata710\\traindata625\\estK'
# filename1 = 'D:\\DATA\\traindata710\\traindata625\\label'
filename = 'test2/EstimateKernel'
filename1 = 'test2/label'



def datagenerator(data_dir = filename,verbose=False):
    file_list = glob.glob(data_dir+'/*.tif')
    KERNEL= []
    for i in range(len(file_list)):
        a = cv2.imread(file_list[i],0)
        KERNEL.append(a)
    if verbose:
        print(str(i+1) + '/' + str(len(file_list)) + 'is done')
    KERNEL = np.array(KERNEL, dtype='uint8')                ##将图像转化为灰度图后才不报错
    KERNEL = np.expand_dims(KERNEL, axis=3)
    discard_n = len(KERNEL) - len(KERNEL) // batch_size * batch_size  # because of batch namalization
    KERNEL = np.delete(KERNEL, range(discard_n), axis=0)
    print('training data finished')
    return KERNEL

# def datageneratorLABEL(data_dir = filename, data_dir1 = filename1, verbose=False):
#     file_list = glob.glob(data_dir+'/*.tif')
#     kt = cv2.imread(data_dir1, 0)
#     KT= []
#     for i in range(len(file_list)):
#         KT.append(kt)
#     if verbose:
#         print(str(i+1) + '/' + str(len(file_list)) + 'is done')
#     KT = np.array(KT, dtype='uint8')
#     KT = np.expand_dims(KT, axis=3)
#     discard_n = len(KT) - len(KT) // batch_size * batch_size  # because of batch namalization
#     KT = np.delete(KT, range(discard_n), axis=0)
#     print('^_^-training data finished-^_^')
#     return KT




class DATASET(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): fake data
    """
    def __init__(self, DATA, LABEL):
        super(DATASET, self).__init__()
        self.DATA = DATA
        self.LABEL = LABEL


    def __getitem__(self, index):
        # if self.batch_x is not None:
        batch_x = self.DATA[index]
        # if self.batch_y is not None:
        batch_y = self.LABEL[index]          #这步没有问题
        return batch_x, batch_y

    def __len__(self):
        return self.DATA.size(0)




# if __name__ == '__main__':
#
#     data = datagenerator(data_dir=filename)
#     KT = datagenerator(data_dir=filename1)

    # RS = KT - data
