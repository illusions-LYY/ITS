# -*- coding:utf-8 -*-
import jieba
import re
import pandas as pd
import numpy as np
import multiprocessing as mps

"""
文本前处理整体思路：
    -1 将全部数据拉成一个string，正则匹配全部HTML标签并删除，返回大string；
    -2 输入大string，正则匹配Rt//triangel和定义新运算的LaTeX，并替换；
    -3 将大string按照"[内容分隔]"分隔，形成一个Series；
    -4 按每道题的顺序正则匹配全部k,b,a,b,c,并替换每道题中第一次出现、以及随后出现的每一次上述字母；
    -5 jieba分词，输出最终的结果(另外的py文件实现)
    
    :input:有id、body、explains列的DataFrame
    :return:有id、body、explains列的处理过的DataFrame
"""


def split_data(data):
    """
    设cpu数量是m，将数据等分成m块
    param data: 待分数据
    return: 被分成m块的data数据，形式为list
    """
    cores = mps.cpu_count()
    split_num = np.linspace(0, len(data), cores + 1, dtype=int)
    data_seg = [data[split_num[j]:split_num[j + 1]] for j in range(len(split_num) - 1)]
    return data_seg


def purify(s, r_h):
    """
    专门用于清理HTML标签，得出我们想要的处理后的题目（多进程的子任务）
    param s: 待处理字符串
    param r_h: HTML标签集合
    return: 删除HTML标签后的字符串
    """
    for e in r_h:
        s = s.replace(e, ' ')
    return s


def html_process(data):
    """
    文本前处理：找到所有HTML标签，并悉数删除
    param data: DataFrame，必含有body和explains列
    return: 删除了HTML标签的、展平成一个string的字符串
    """
    qqq = data['body'] + ' [内容分隔] ' + data['explains']
    bd_ex = ''
    for i in qqq:
        i = i + ' [题目分隔] '
        bd_ex += i
    # 将全部题目文本连接成一个长字符串，便于正则直接处理、替换，不做重复事情（可以优化写成同步的）
    pattern_html = re.compile(r'<[^(0-9)][^(<,>,，,$)]*>')
    # 开头第一位是“<”，第二位不能是数字，紧接着匹配任意次的任意字符，但不可以是(<,>,，,$)，最后末尾匹配“>”
    result_html = pattern_html.findall(bd_ex)
    trash = ['< \\\\\\\\frac{a+1}{2} \\\\\\\\\\\\\\\\ x>',
             '< \\\\\\\\frac{m+1}{2} \\\\\\\\\\\\\\\\ x>',
             '< \\\\\\\\frac{m+4}{2} \\\\\\\\\\\\\\\\ x>',
             '<-1\\\\5x-2b>',
             '<-1\\\\\\\\\\\\\\\\5x-2b>',
             '<\\\\\\\\frac{a+2}{3} \\\\\\\\\\\\\\\\ x>',
             '<\\\\\\\\frac{a-1}{4} \\\\\\\\\\\\\\\\ x>',
             '<a+1\\\\\\\\\\\\\\\\ x>',
             '<a+4\\\\\\\\\\\\\\\\ x>',
             '<m+2\\\\\\\\\\\\\\\\ x>',
             '<m-6\\\\\\\\\\\\\\\\ x>',
             '<m-7\\\\\\\\\\\\\\\\ x>',
             '<m-9 \\\\\\\\\\\\\\\\ x>']
    # 这些是匹配到的非HTML标签，不管也罢，最终分词终究被删除
    r_html = sorted(set(result_html))
    res_html = sorted(set(r_html) - set(trash))  # 最终的全部HTML标签

    # 使用多进程处理缓慢的HTML替换过程
    cpus = mps.cpu_count()
    pool = mps.Pool(processes=cpus)
    seg_bd_ex = split_data(bd_ex)
    r = []
    for i in seg_bd_ex:
        i_ex = pool.apply_async(purify, args=(i, res_html))
        r.append(i_ex)

    pool.close()
    pool.join()

    res = [j.get() for j in r]  # 被分成了4段的大string
    return ''.join(res)


def exchange_rt_latex(ss):
    """
    文本前处理：替换定义新运算的LaTeX；替换Rt为直角三角形；替换平行符号&平行LaTeX；替换backsim→相似。
    param ss: 还未替换的大string
    return: 依次替换后的大string
    """
    new_cal = ['diamondsuit', 'circledcirc', 'bigstar', 'boxtimes',
               'heartsuit', 'otimes', 'spadesuit', 'star', 'yen']
    for l in new_cal:
        ss = ss.replace(l, 'clubsuit')

    prt = re.compile(r'Rt ?}?\${0,2} {0,3}\\{0,8}triangle')
    rt = set(prt.findall(ss))
    for w in rt:
        ss = ss.replace(w, '直角三角形')

    parallel = ['//', '∥', 'parallel']
    for p in parallel:
        ss = ss.replace(p, '平行')

    # 替换相似、垂直
    ss = ss.replace('backsim', '相似')
    ss = ss.replace('perp', '垂直')

    return ss


def str_to_series(s):
    """
    将大string还原成Series
    param s:大string
    return: Series格式的、未将body和explains分开的每道题目
    """
    s = s.split(' [题目分隔] ')  # 按照题目分隔将题目分开
    s.pop()
    return pd.Series(s)


def series_to_df(series, data):
    """
    将series还原成DataFrame（原data格式）
    param series: input
    param data: 原始数据，其实就是提供problem_id，以便将做好前处理的题目文本按原有id放回
    return: 完全处理过后的data，可以进行下一步的jieba分词
    """
    b_ = []
    e_ = []
    for idx, item in enumerate(series):
        b_e = item.split(' [内容分隔] ')
        b_.append(b_e[0])
        e_.append(b_e[1])
    # 按照【内容分隔】将body、explains分开
    data['body'] = pd.Series(b_)
    data['explains'] = pd.Series(e_)
    return data


"""
匹配一次函数y=kx+b的逻辑：change_bk/find_linear/linear_change
    -1 在每一道题目文本中匹配y=kx+b（类似模式）；
    -2 若该题匹配到，则替换匹配到的位置以后的全部k&b；
    -3 注意，防止将body/backsim等固定词语的"b"一并替换成了"截距"，因而我们要提前转换大小写，并规定正则只匹配小写字母。
"""


def change_bk(sentence, dic):
    """
    将题目中所有LaTeX单词中的小写k，b改为大写K，B，防止替换时将body的'b'也替换成"截距"
    param sentence:题目文本
    param dic:转换词典——转换过去后还要再转换回来，所以key-value二者对调即可反转该函数的功能
    return:修改过的题目文本
    """
    for word in list(dic.keys()):
        sentence = sentence.replace(word, dic[word])
    return sentence


def find_linear(ss):
    """
    判断一段文本中是否含有y=kx+b这样的模式。若有，则返回k，b模式及其位置；若无，返回'not linear'
    param ss: 题目文本
    return k/b:匹配到的k和b（可能是k_1这种形式）；
    return loc[0]:匹配到的一次函数解析式开头——再此处以后，所有k，b都替换为"斜率、截距"
    """
    p_linear = re.compile(r'y(?:_)?[^$]{0,3}=(?: )?(?:\-)?k(?:_)?[^$]{0,3}x(?:\+|\-)b(?:_)?[\w]{0,3}')
    loc = p_linear.search(ss)
    if loc is None:
        return 'not linear'
    loc = loc.span()
    linear_ss = ss[loc[0]:loc[1]]
    k = linear_ss[linear_ss.find('k'):linear_ss.find('x')]
    b = linear_ss[linear_ss.find('b'):]
    return k, b, loc[0]  # 最后一个参数是匹配到的第一个y=kx+b的结束为止，此后的k、b都换成“斜率，“截距”


def linear_change(series):
    """
    专门用于将固定题目中，固定位置的k&b替换掉
    param series:传入的是全部题目body+explains形式，pandas.Series格式；
    return: 替换后的结果，仍是Series格式
    """
    dic_bk = {'body': 'Body', 'backsim': 'Backsim', 'because': 'Because', 'begin': 'Begin', 'beta': 'Beta',
              'bigstar': 'Bigstar', 'boxtimes': 'Boxtimes', 'clubsuit': 'cluBsuit', 'checkmark': 'checKmarK'}
    reverse_dic_bk = {v: k for k, v in dic_bk.items()}

    for idx, item in enumerate(series):
        res = find_linear(item)
        if res != 'not linear':
            k, b = res[0], res[1]
            tail = change_bk(item[res[2]:], dic_bk)  # 匹配上了y=kx+b结构，则在替换k&b前将带字母k、b的关键字换成K、B
            tail = tail.replace(k, '斜率')
            tail = tail.replace(b, '截距')
            tail = change_bk(tail, reverse_dic_bk)  # 再变回来，。。。。
            series[idx] = item[:res[2]] + tail
    return series


"""
匹配二次函数y=ax^2+bx+c的逻辑：find_quadratic/quadratic_change
    -1 在每一道题目文本中匹配y=ax^2+bx+c（类似模式）；
    -2 对于匹配到的题目，只修改匹配到的部分（即解析式当中的）a&b，不再向后匹配；
    -3 尝试匹配"c"，匹配不到就放弃。
"""


def find_quadratic(ss):
    """
    判断一段文本中是否含有y=ax^2+bx+c这样的模式。若有，则返回c的模式及二次函数解析式模式的位置；若无，返回'not quadratic'
    param ss:题目文本；
    return:c;匹配之处的开头和结尾。
    """
    p_quadratic = re.compile(r'ax\^2(?:\+|\-)bx(?:\+|\-)(?:c)?')
    loc = p_quadratic.search(ss)
    if loc is None:
        return 'not quadratic'
    loc = loc.span()
    c = 'c' if ss[loc[1] - 1] == 'c' else None
    return c, loc[0], loc[1]


def quadratic_change(series):
    """
    替换匹配到的a,b,c
    param series: 全部题目，数据结构为pandas.Series
    return: 替换后的全部题目，数据结构为pandas.Series
    """
    for idx, item in enumerate(series):
        res = find_quadratic(item)
        if res != 'not quadratic':
            motif = item[res[1]:res[2]]
            motif = motif.replace('a', '二次项系数')
            motif = motif.replace('b', '一次项系数')
            if res[0]:
                motif = motif.replace('c', '常数项')
            series[idx] = item.replace(item[res[1]:res[2]], motif)
    return series


# 主函数，完整地处理文本调用它！
def pre_processing(data_df):
    # 删除HTML标签
    big_str_0 = html_process(data_df)
    # 替换Rt→直角三角形，替换新定义相关的LaTeX，替换平行
    big_str_1 = exchange_rt_latex(big_str_0)
    # 将大string转成Series
    ser1 = str_to_series(big_str_1)
    # 替换一次函数的k,b
    ser2 = linear_change(ser1)
    # 替换二次函数的abc
    ser3 = quadratic_change(ser2)
    # 还原成原本的DataFrame模样，返回DataFrame
    data_processed = series_to_df(ser3, data_df)

    return data_processed
