# -*- coding: utf-8 -*-
# @Time    : 2019/2/13 11:00
# @Author  : HuangHao
# @Email   : 812116298@qq.com
# @File    : test.py
import os
from scipy.io import loadmat

for maindir, subdir, file_name_list in os.walk("./dataset"):
        for filename in file_name_list:
            file = loadmat("./dataset/{0}".format(filename))['P1']
            print(filename+":"+str(len(file)))