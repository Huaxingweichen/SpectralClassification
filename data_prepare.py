# -*- coding: utf-8 -*-
# @Time    : 2019/2/1 11:39
# @Author  : HuangHao
# @Email   : 812116298@qq.com
# @File    : data_prepare.py
from scipy.io import loadmat
import torch.utils.data as Data
import numpy as np

import torch

class SpectralData:
    """create dataset from ./dataset/*.mat"""
    def __init__(self):
        self.data_name = ['M0_15_5000', 'M1_15_5000', 'M2_15_5000', 'M3_15_5000', 'M4_15_5000']
        self.train_test_rate = 0.8 # the rate of train set and test set in all

    def dataPrepare(self):
        print('==> loading data..')
        x_train = None
        y_train = None
        x_test = None
        y_test = None
        for name in self.data_name:
            data = loadmat("dataset/{0}.mat".format(name))['P1']
            traindata_x = torch.from_numpy(data[:int(self.train_test_rate*len(data))]).type(torch.FloatTensor)
            traindata_y = torch.from_numpy(np.array([self.data_name.index(name) for i in range(len(traindata_x))])).type(torch.LongTensor).view(-1,1)
            traindata_y = torch.zeros(len(traindata_y), 5).scatter_(1, traindata_y, 1).type(torch.FloatTensor)

            testdata_x = torch.from_numpy(data[int(self.train_test_rate*len(data)):]).type(torch.FloatTensor)
            testdata_y = torch.from_numpy(np.array([self.data_name.index(name) for i in range(len(testdata_x))])).type(torch.LongTensor).view(-1,1)
            testdata_y = torch.zeros(len(testdata_y), 5).scatter_(1, testdata_y, 1).type(torch.FloatTensor)

            if x_train is None:
                x_train = traindata_x
                y_train = traindata_y
                x_test = testdata_x
                y_test = testdata_y
            else:
                x_train = torch.cat((x_train, traindata_x), dim=0)
                y_train = torch.cat((y_train, traindata_y), dim=0)
                x_test = torch.cat((x_test, testdata_x), dim=0)
                y_test = torch.cat((y_test, testdata_y), dim=0)
        x_train = x_train.view(-1,5000)
        x_test = x_test.view(-1,5000)
        train_dataset = Data.TensorDataset(x_train, y_train)
        test_dataset = Data.TensorDataset(x_test, y_test)

        print('==> data loading finished..')
        return train_dataset, test_dataset


