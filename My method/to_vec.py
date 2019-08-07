# -*- coding:utf-8 -*-

import model_utils
import pandas as pd
import pickle

data = pd.read_csv('/Users/shen-pc/Desktop/WORK/ITS/My method/results/problem_0528_jieba.csv')
res_vector = {}
for i in data['id']:
    res_vector[i] = model_utils.get_problem_vector(i)

with open('all_vectors.pkl', 'wb')as f:
    pickle.dump(res_vector, f)
