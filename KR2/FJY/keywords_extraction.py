
# coding: utf-8

import pandas as pd
import numpy as np
from gensim import models,corpora
import os

def read_keywords(path = './data/wd_list之真的不改了.txt'):
    """
    读取关键词词典
    input：词典路径
    output：根据关键词长度排序的列表
    """
    wd = [i.strip() for i in open(path)]
    wd_sorted = sorted(wd,key = lambda x:len(x),reverse = True)
    return wd_sorted



def takewd(sentence):
    """
    从文本中提取关键词,根据关键词长度依次提取和删除
    input：一个题目文本信息
    output：题目关键词列表
    """
    sen_vec = []
    wordlist = read_keywords()
    for i in wordlist:
        if i in sentence:
            c = sentence.count(i)
            for _ in range(c):
                sen_vec.append(i)
            sentence = sentence.replace(i,'')
    return sen_vec           


def text_kw_list(text):
    """
    题目列表转换关键词
    input：题目文本，列表套列表
    output：模型输入形式，即列表套列表
    """
    if type(text) == list:
        text_vec = []
        for i in text:
            text_vec.append(takewd(i))
        return text_vec
    else:
        return [takewd(text)]
    
def lda_train(problem_set):
    '''
    训练模型
    problem_set:dataframe，colnames = ['id','text']
    return：lda模型
    save:lda模型，dictionary，话题关键词权重矩阵，topic list
    '''
    text_list = list(problem_set['text'])
    text_vec = text_kw_list(text_list)
    dictionary = corpora.Dictionary(text_vec)
    corpus = [dictionary.doc2bow(i) for i in text_vec]
    lda_model = models.ldamodel.LdaModel(corpus,id2word=dictionary,num_topics=150,passes=20)
    save_something(lda_model,dictionary)
    return lda_model

def save_something(lda_model,dictionary):
    '''
    保存模型信息到磁盘：lda模型，dictionary，话题关键词权重矩阵，topic list
    '''
    #保存模型
    model_path = './model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    dictionary.save('./model/dicitionary.dic')
    lda_model.save('./model/lda')
    #话题关键词权重矩阵
    mat = lda_model.get_topics()
    pd.to_pickle(mat,'./data/关键词权重矩阵.pkl')
    #保存话题
    topics_list = lda_model.print_topics(num_topics=-1)
#     f = open('./data/topics_list','w')
#     for i in topics_list:
#         f.write(str(i))
#         f.write('\n')
#     f.close()  
    pd.to_pickle(topics_list,'./data/topic_list.pkl')
    print('save lda_model,dictionary,topic-terms matrix,topic list')
    
def train_increasing(problem_set,save = False):
    '''
    模型增量训练，需要多个样本
    problem_set:dataframe，colnames = ['id','text']
    return：新lda模型
    save：lda模型，dictionary，话题关键词权重矩阵，topic list
    '''
    other_texts = list(problem_set['text'])
    lda = models.LdaModel.load('./model/lda')
    dictionary = corpora.Dictionary.load('./model/dicitionary.dic')
    text_vec = text_kw_list(other_texts)
    dictionary.add_documents(text_vec)
    other_corpus = [dictionary.doc2bow(text) for text in text_vec]
    lda.update(other_corpus)
    if save == True:        
        save_something(lda,dictionary)
    
    return lda
      
def get_topic(problem_set):
    """
    基于已经保存的模型提取题目中的话题，预测部分
    problem_set:dataframe，colnames = ['id','text']
    return：topic列表，形如[(1,0,876),(100,0.123)]"""
    text = list(problem_set['text'])
    text_vec = text_kw_list(text)
    lda = models.LdaModel.load('./model/lda')
    dictionary = corpora.Dictionary.load('./model/dicitionary.dic')
    topics = []
    for i in text_vec:
        text_bow = dictionary.doc2bow(i)
        topics.append(list(lda[text_bow]))
    return topics

def to_matrix(problem_set):
    '''
    输出题目-话题权重矩阵
    problem_set:dataframe，colnames = ['id','text']
    return：mat->array，150列的矩阵，每行为一个题目,列为150个topic
            id2index->dict，键为题目id，值为矩阵行号
            id2topic->dict, 键为topic编号，值为topic内容
    '''
    id2index = {}
    id2topic = {}
    pid = list(problem_set['id'])
    new_text_topic = get_topic(problem_set)
    mat = np.zeros([len(new_text_topic),150])
    for i in range(len(new_text_topic)): 
        id2index[i] = pid[i]
        for index,wei in new_text_topic[i]:
            mat[i][index] = wei   
    topic_list = pd.read_pickle('./data/topic_list.pkl')
    for i in topic_list:
        id2topic[i[0]] = i[1]
    return mat,id2index,id2topic

if __name__ == '__main__':
    #模型训练
    df = pd.read_csv('./data/last_question.csv')
    df['text'] = df.body+df.explains+df.goal_name
    problem_set = pd.concat([df['id'],df['text']],axis=1)
    lda_train(problem_set)
