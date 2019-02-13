# -*- coding: utf-8 -*-
# @Time    : 2019/2/1 10:58
# @Author  : HuangHao
# @Email   : 812116298@qq.com
# @File    : networks.py

import torch
import torch.nn as nn
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 模型需继承nn.Module
class VGG(nn.Module):
# 初始化参数：
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.input = nn.Linear(5000, 1024)
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(16384, 5)

# 模型计算时的前向过程，也就是按照这个过程进行计算
    def forward(self, x):
        out = self.input(x).view(-1, 1, 1024)
        out = self.features(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm1d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool1d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net)
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.input = nn.Linear(5000, 1024)
        self.conv = self._make_layers()
        self.output = nn.Linear(64*1024, 5)

    def forward(self, x):
        out = self.input(x).view(-1, 1, 1024)
        out = self.conv(out).view(out.size(0), -1)
        out = self.output(out)

        return out

    def _make_layers(self):
        layers = []
        layers += [nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)]
        layers += [nn.Conv1d(64, 64, kernel_size=3, padding=1),
                   nn.BatchNorm1d(64),
                   nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)


class MyDNN(nn.Module):
    def __init__(self):
        super(MyDNN, self).__init__()
        self.input = nn.Linear(5000, 1024)
        self.dnn1 = nn.Linear(1024, 512)
        self.dnn2 = nn.Linear(512, 256)
        self.dnn3 = nn.Linear(256, 128)
        self.dnn4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 5)

    def forward(self, x):
        out = self.input(x)
        out = self.dnn1(out)
        out = self.dnn2(out)
        out = self.dnn3(out)
        out = self.dnn4(out)
        out = self.output(out)

        return out

