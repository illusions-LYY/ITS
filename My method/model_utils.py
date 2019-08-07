# -*- coding:utf-8 -*-
import os
from gensim.models import doc2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

"""
    -1 本文件提供doc2vec的model训练、model"伪增量"训练、获取题目文本向量、
        计算任意两个向量之间的余弦相似度、"从题库中"获取最相似题目等操作。
    -2 对于增量题目，建议还是加上老题库一同重新训练模型。对于增量题目，infer_vector()转换得到的题目文本向量精度有限。
    -3 当存在增量题目，并且想将增量题目与老题目一起进行model训练，则删除/results/vectors.kv这个model文件，
        并将/results文件夹下所有.csv文件与处理后的增量题目合并成一个.csv文件，再调用train()方法即可。
    -4 注意：原数据必须有'id'列，且problem_id一定要保持唯一性。
    
"""


def train(data):
    """
    若model不存在，则以输入的data为全体样本进行model训练；
    若model已存在，则load进来model后，使用输入的增量data进行doc2vec神经网络权重更新。
    :param data:DataFrame，必有"cut"列，训练doc2vec的原始数据
    :return:训练好的doc2vec model
    """
    model_path = "./results/vectors.kv"
    abs_path = os.path.abspath('.')
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # 使用problem_id作为题目索引
    document = [doc2vec.TaggedDocument(doc, [pid]) for num, (pid, doc) in data[['id', 'cut']].iterrows()]
    if os.path.isfile(model_path):  # 如果model已存在，则进行增量训练，更新doc2vec神经网络权重
        print("====================模型已存在，开始进行增量训练====================")
        model = incremental_train(model_path, document)
        if os.path.isfile('./results/data_incremental.csv'):  # 将数据保存到本地（增量数据单独存放，独自成一个文件）
            data_incremental = pd.read_csv('./results/data_incremental.csv', index_col=0)
            data = pd.concat([data_incremental, data])
        print('原始数据量：', model.corpus_count, '\n', '增量数据量', len(data))
        data.to_csv('./results/data_incremental.csv')
    else:  # 如果model不存在，直接进行训练
        print("====================模型不存在，开始进行初始训练====================")
        model = doc2vec.Doc2Vec(document, vector_size=256, window=10, min_count=2, workers=4,
                                alpha=0.025, min_alpha=0.025, epochs=20)
        data.to_csv('./results/data_origin.csv')  # 将数据保存到本地
    fname = get_tmpfile(abs_path + model_path.replace('.', '', 1))  # 特别注意，这里必须要填绝对路径
    model.save(fname)  # 将model保存到本地
    return model


def incremental_train(model_path, document):
    """
    doc2vec理论上无法进行增量训练，所谓"增量训练"，只不过是通过新的文本更新神经网络参数，这与其他语言模型如LDA不同；
    因而该方法实际上只更新神经网络权重，意图让我们在infer_vector的时候更准确一些，
    但这并不会更改词库或题目文本库。所以当题库中新题数目达到一定数量的时候，应该再次使用train方法进行整体训练。
    :param document:
    :param model_path:已有model的路径
    :return:
    """
    model = KeyedVectors.load(model_path)
    model.train(documents=document, total_examples=model.corpus_count + len(document), epochs=20)
    return model


def get_problem_vector(pid):
    """
    通过题目id获取题目向量。其中，未在原始题库中的题目，是通过infer_vector()方法获得的文本向量。
    :param pid: str格式，题目id
    :return: numpy.array格式，256维embedding向量
    """
    model_path = "./results/vectors.kv"
    model = KeyedVectors.load(model_path)
    try:
        vec = model.docvecs[pid]
    except KeyError:
        print("这里有一道新题", pid)
        # 新题应先进行前处理、分词、模型训练等动作，故定会被存在data_incremental.csv文件中。读取其pid对应的分词结果
        incre_data = pd.read_csv('./results/data_incremental.csv', index_col=0)
        words = incre_data[incre_data['id'] == pid]['cut'].tolist()[0].split()
        vec = model.infer_vector(words)  # 推测ssh
    return vec


def cal_cos_sim(p1, p2):
    """
    计算余弦相似度——使用gensim库已封装的函数。若有新题存在，则调用自己的my_cos_sim()方法计算相似度。
    :param p1: 题目向量1的problem_id
    :param p2: 题目向量2的problem_id
    :return: 余弦相似度
    """
    model_path = "./results/vectors.kv"
    model = KeyedVectors.load(model_path)
    try:
        sim = model.docvecs.similarity(p1, p2)
    except KeyError:
        vector1 = get_problem_vector(p1)
        vector2 = get_problem_vector(p2)
        sim = __my_cos_sim(vector1, vector2)
    return sim


def __my_cos_sim(vector_a, vector_b):
    """
    私有方法：计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


def most_similar(pid, topn=10):
    """
    计算在题库中与目标题目最相近的题目。
    :param pid: 作为正面典型出现的题目id
    :param topn: 列出前n名
    :return: 最相近的前n个题目文本，相似度降序排列
    """
    model_path = "./results/vectors.kv"
    model = KeyedVectors.load(model_path)
    try:
        res = model.docvecs.most_similar(pid, topn=topn)
    except(KeyError, TypeError):
        vec = get_problem_vector(pid)
        res = model.docvecs.most_similar([vec], topn=topn)
    return res
