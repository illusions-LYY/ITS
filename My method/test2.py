# -*- coding:utf-8 -*-

import model_utils
import utils
import jieba_cut
import pandas as pd

data = pd.read_csv('/Users/shen-pc/Desktop/WORK/ITS/My method/results/problem_0528_jieba.csv', index_col=0)
sim1 = model_utils.cal_cos_sim(data.iloc[5322]['id'], data.iloc[5323]['id'])
print('原model的结果[5322 vs 5323]=', sim1)
# 来几道新题试一下
data_new = pd.read_csv('/Users/shen-pc/Desktop/WORK/ITS/data/real_item.csv', index_col=0)
data_new.rename(columns={'problem_id': 'id'}, inplace=True)
data_new = data_new.loc[:20]
data_new = utils.pre_processing(data_new)
data_new = jieba_cut.cut(data_new)
# 加入新数据进行训练：
model_new = model_utils.train(data_new)
sim2 = model_utils.cal_cos_sim(data.iloc[5322]['id'], data.iloc[5323]['id'])
sim3 = model_utils.cal_cos_sim(data.iloc[100]['id'], data.iloc[1000]['id'])
print('新model的结果[5322 vs 5323]=', sim2)
print('新model的结果[100 vs 1000]=', sim3)
# 涉及原本没有的题目：
sim4 = model_utils.cal_cos_sim(data.iloc[0]['id'], data_new.iloc[0]['id'])
print('新model的结果[old 0 vs new 0]=', sim4)

# 最相似:
most1 = model_utils.most_similar(data.iloc[0]['id'])
most2 = model_utils.most_similar(data_new.iloc[0]['id'])
print('\n\n\n\n', data.loc[0, 'cut'], '\n', most1, '\n\n')
print(data_new.loc[0, 'cut'], '\n', most2)
