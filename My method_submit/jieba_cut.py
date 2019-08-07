# -*- coding:utf-8 -*-
import pandas as pd
import jieba
import re

"""
以下为jieba分词部分：
    -1 先将前处理完毕的题目文本（body+explains）加上二级目标和subseciton（如果有的话）;
    -2 jieba分词，同时对停用词表加以筛选。注意，jieba分词的时候要再如我预先设置好的userdict；
    -3 分词完毕，输出原DataFrame，并新加一列"cut"，为分词结果。
    
    :input:有id、body、explains列的DataFrame
    :return:有id、body、explains、cut列的处理过的DataFrame。若原DataFrame有'goal_name'列，则会多出section、subsection列
"""


def read_txt(path):
    lines = []
    with open(path, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip()
            lines.append(line)
    return lines


def find_section_subsection(data):
    data['section'] = ''
    data['subsection'] = ''
    section = read_txt('./dependencies/sections.txt')
    subsection = read_txt('./dependencies/subsections.txt')
    for ids, s in enumerate(data['goal_name']):
        if not is_number(s[0]):
            signal_section = s[:s.find('.')].replace('-', '.')  # 6.1
            for i in section:
                if signal_section == i[:len(signal_section)]:
                    data.loc[ids, 'section'] = i[len(signal_section):]
            subsection_signal = s[:s[len(signal_section) + 1:].find('.') + len(signal_section) + 1].replace('-', '.')
            for j in subsection:
                if subsection_signal == j[:len(subsection_signal)]:
                    data.loc[ids, 'subsection'] = j[len(subsection_signal):]
    return data


def add_goal(data):
    data['explains'] = data['explains'] + data['goal_name']
    return data


def add_section_subsection(data):
    data['explains'] = data['explains'] + data['section'] + ',' + data['subsection'] + ','
    return data


def is_number(s):
    """
    判断一个字符串是不是数字。注：若是，返回False；不是，返回True（反着的）
    param s: 字符串
    return: True or False
    """
    try:
        float(s)
        return False
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return False
    except (TypeError, ValueError):
        pass

    return True


def seg_words(ss):
    """
    将经前处理（HTML、替换）的文本进行jieba分词，并将删除自定义的停用词
    param ss: 待分词的字符串己定义的不可删除的英文LaTeX单词,list
    return:分好词的题目文本，以空格分隔
    """
    # 载入用户自定义词典(my_list)、自定义不可删除的LaTeX单词(user_latex)、停用词(stopwords)
    my_jieba = read_txt("./dependencies/my jieba list.txt")
    jieba.load_userdict(my_jieba)
    user_latex = my_jieba[-12:]
    stopwords = read_txt('./dependencies/my stop list.txt')

    # 正则匹配字符串中的代数式or字母，大小写均匹配，然后删除这些字母中我们想保留的LaTeX关键词
    pattern = re.compile(r'[\d]*[a-zA-Z]+')
    letter = pattern.findall(ss)
    letter_shelter = list(set(letter) - set(user_latex))

    # jieba分词
    ss_list = jieba.cut(ss.strip())
    outstr = ''
    for word in ss_list:
        word_s = word.strip()
        stop = set(stopwords + letter_shelter)
        # 这么多if：word在停用词表里、在正则找到的代数式or字母里、是空串和制表符、是数字的，均删除
        if word_s not in stop:
            if word_s != '' and word_s != '\t' and is_number(word_s):
                outstr += word_s
                outstr += " "

    return outstr


# 主函数，完整地jieba分词过程调用它！
def cut(data):
    """
    param data: 全部题目的数据表，DataFrame格式，必有problem_id/body/explains三列，goal_name、subsection列最好也有，有助于提升计算题目文本相似度的精确度
    return: 全部题目的数据表，DataFrame格式，多了"cut"列
    """
    if 'goal_name' in data.columns:
        if 'section' not in data.columns or 'subsection' not in data.columns:
            data = find_section_subsection(data)  # 寻找对应的章节小节名
        data = data.fillna('')
        data = add_section_subsection(data)  # 添加章节小节名到explains列
    data = add_goal(data)  # 添加二级目标到explains列
    data['together'] = data['body'] + data['explains']

    cut_list = []
    for i in data['together']:
        cut_list.append(seg_words(i))
    data['cut'] = pd.Series(cut_list)
    return data
