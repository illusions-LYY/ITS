# -*- coding:utf-8 -*-

import os
import utils
import jieba_cut
import pandas as pd
import time
import model_utils

if not os.path.exists('./results'):
    os.makedirs('./results')
t0 = time.time()
print('============================文本前处理开始============================')
data = pd.read_csv('/Users/shen-pc/Desktop/WORK/ITS/KR2/LSD_data/problem_0528.csv')
data_proc = utils.pre_processing(data)
data_proc.to_csv('./results/problem_0528_preprocessing.csv')
t1 = time.time()
print('文本前处理耗时：', (t1 - t0) / 60, 'min')
print('============================文本前处理over============================', '\n\n')

'''
------------------------------------------------------------------------------------------------------------------------
'''

print('============================分词开始============================')
data_cut = jieba_cut.cut(data_proc)
data_cut.to_csv('./results/problem_0528_jieba.csv')
t2 = time.time()
print('分词耗时：', (t2 - t1) / 60, 'min')
print('============================分词over============================', '\n\n')

'''
------------------------------------------------------------------------------------------------------------------------
'''

print('============================model训练开始============================')
d2v = model_utils.train(data_cut)
t3 = time.time()
print('model训练耗时：', (t3 - t2) / 60, 'min')
print('============================model训练over============================', '\n\n')

'''
------------------------------------------------------------------------------------------------------------------------
'''
print('\n\n\n\n')
print('共耗时', (time.time() - t0) / 60, 'min')
