# coding = utf-8
# author = xy

import json
from rouge import Rouge
import os
import pandas as pd
import numpy as np
import copy
import jieba
import sys
import re
import logging
import gensim
from gensim.models.word2vec import LineSentence
from sklearn import model_selection
from config import config_base
import vocab
config = config_base.config


# convert .json to .pandas
# return: df
def organize_data(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
        result = []
        for dc in data:
            temp = [dc['article_id'], dc['article_type'], dc['article_title'], dc['article_content']]
            for items in dc['questions']:
                r = copy.deepcopy(temp)
                r = r + list(items.values())
                result.append(r)
        if 'answer' in data[0]['questions'][0]:
            columns = ['article_id', 'article_type', 'article_title', 'article_content', 'question_id',
                       'article_question', 'article_answer', 'question_type']
        else:
            columns = ['article_id', 'article_type', 'article_title', 'article_content', 'question_id',
                       'article_question']
        df = pd.DataFrame(data=result, columns=columns)

        return df


# 1. 除掉多余空格、删除'\u3000'
# 2. 繁简体转换
# 3. 删除答案中，不好的结尾符
def deal_data(df):
    def deal(data):
        result = []
        for i in data:
            i = re.sub(r'\u3000', '', i)
            i = re.sub(r'\s+', ' ', i)

            # 繁简体转换
            i = re.sub(r'０', '0', i)
            i = re.sub(r'１', '1', i)
            i = re.sub(r'２', '2', i)
            i = re.sub(r'３', '3', i)
            i = re.sub(r'４', '4', i)
            i = re.sub(r'５', '5', i)
            i = re.sub(r'６', '6', i)
            i = re.sub(r'７', '7', i)
            i = re.sub(r'８', '8', i)
            i = re.sub(r'９', '9', i)
            i = re.sub(r'．', '.', i)

            # 去除前后空格
            i = i.strip()

            result.append(i)
        return result

    # 1, 2
    df['title'] = deal(df['article_title'].values)
    df['content'] = deal(df['article_content'].values)
    df['question'] = deal(df['article_question'].values)

    # 3
    if 'article_answer' in df:
        df['answer'] = deal(df['article_answer'].values)
        answers = df[df['answer'] != '']['answer'].values
        drop_list = ['。', '，', '：', '！', '？']
        answers = [answer[:-1].strip() if answer[-1] in drop_list else answer for answer in answers]
        df.loc[df['answer'] != '', 'answer'] = answers

    return df


# shorten content
def shorten_content_all(df, max_len):
    """
    :param df:
    :param max_len:
    :return: df
    """
    sys.setrecursionlimit(1000000)
    rouge = Rouge(metrics=['rouge-l'])

    def match(title, content, question, max_len):

        # 如果无法用'。'划分，直接返回 title
        if '。' not in content:
            return title

        def count(flag, content_list):
            """ 查数 """
            number = 0
            for i in range(len(flag)):
                if flag[i] != 0:
                    number += len(content_list[i])+1
            return number

        # 过滤
        content_list = content.split('。')
        temp = []
        for c in content_list:
            if c not in ['', ' ', '  ']:
                temp.append(c)
        content_list = temp
        content_list = [jieba.lcut(c) for c in content_list]
        content_len = len(content_list)

        question_str = ' '.join(jieba.lcut(question))

        # 相似性得分: rouge-l
        scores = []
        for c in content_list:
            if ''.join(c) in question:
                scores.append(-5)
                continue
            c_str = ' '.join(c)
            score = rouge.get_scores(c_str, question_str, avg=True)['rouge-l']['r']
            scores.append(score)

        # 标记类型
        flag = np.zeros(content_len)
        title_number = len(jieba.lcut(title))
        max_len = max_len - title_number
        flag_result = flag.copy()

        # 核心句:
        max_score = max(scores)
        for i in range(content_len):
            if scores[i] == max_score:
                flag[i] = -1
        number = count(flag, content_list)
        if number <= max_len:
            flag_result = flag.copy()
        else:
            question_set = set(jieba.lcut(question, cut_all=True))
            for j in range(content_len):
                if flag[j] == -1:
                    c = ''.join(content_list[j])
                    c_set = set(jieba.lcut(c, cut_all=True))
                    s = len(c_set & question_set)
                    flag[j] = s
            max_s = max(flag)
            for j in range(content_len):
                if flag[j] == max_s:
                    flag_result[j] = -1

            temp = []
            c = 0
            for j in range(content_len):
                if flag_result[j] == -1:
                    c += len(content_list[j]) + 1
                    if c > max_len:
                        break
                    temp.append(''.join(content_list[j]))
            result = [title] + temp

            return '。'.join(result)

        # 核心句下一句
        if number <= max_len:
            for i in range(content_len):
                if (flag[i] == -1) and (i+1 < content_len) and (flag[i+1] == 0):
                    flag[i+1] = -2
            number = count(flag, content_list)
            if number < max_len:
                flag_result = flag.copy()

        # 最后一句
        if number < max_len:
            if flag[-1] == 0:
                flag[-1] = -3
            number = count(flag, content_list)
            if number < max_len:
                flag_result = flag.copy()

        # 第一句
        if number < max_len:
            if flag[0] == 0:
                flag[0] = -4
            number = count(flag, content_list)
            if number < max_len:
                flag_result = flag.copy()

        # 蕴含句（上+中+下）
        if number < max_len:
            for i in range(content_len):
                if scores[i] == -5:
                    if (i-1 >= 0) and (flag[i-1] == 0):
                        flag[i-1] = -5
                    if flag[i] == 0:
                        flag[i] = -5
                    if (i+1 < content_len) and (flag[i+1] == 0):
                        flag[i+1] = -5
            number = count(flag, content_list)
            if number < max_len:
                flag_result = flag.copy()

        # 核心句下下句
        if number < max_len:
            for i in range(content_len):
                if (flag[i] == -1) and (i+2 < content_len) and (flag[i+2] == 0):
                    flag[i+2] = -6
            number = count(flag, content_list)
            if number < max_len:
                flag_result = flag.copy()

        # 核心句上一句
        if number < max_len:
            for i in range(content_len):
                if (flag[i] == -1) and (i-1 >= 0) and (flag[i-1] == 0):
                    flag[i-1] = -7
            number = count(flag, content_list)
            if number < max_len:
                flag_result = flag.copy()

        # 核心句下下下句
        if number < max_len:
            for i in range(content_len):
                if (flag[i] == -1) and (i+3 < content_len) and (flag[i+3] == 0):
                    flag[i+3] = -8
            number = count(flag, content_list)
            if number < max_len:
                flag_result = flag.copy()

        # 核心句上上句
        if number < max_len:
            for i in range(content_len):
                if (flag[i] == -1) and (i-2 >= 0) and (flag[i-1] == 0):
                    flag[i-2] = -9
            number = count(flag, content_list)
            if number < max_len:
                flag_result = flag.copy()

        result = [title]
        for i in range(content_len):
            if flag_result[i] != 0:
                result.append(''.join(content_list[i]))

        # 过滤重复
        temp = []
        for r in result:
            if r not in temp:
                temp.append(r)
        result = temp

        return '。'.join(result)

    titles = df['title'].values
    contents = df['content'].values
    questions = df['question'].values

    merge = [match(t, c, q, max_len) for t, c, q in zip(titles, contents, questions)]
    df['merge'] = merge

    # 评估数据集构建效果
    if 'answer' in df:
        answers = df['answer'].values
        is_in = [True if a in m else False for m, a in zip(merge, answers)]
        df['is_in'] = is_in
        print('shorten content, accuracy: %.4f' % (sum(is_in)/len(df)))

    merge_len = [len(jieba.lcut(m)) for m in merge]
    df['len'] = merge_len
    print('max length: %d' % max(merge_len))
    print('min length: %d' % min(merge_len))
    print('mean length:%d' % df['len'].mean())
    print('median length:%d' % df['len'].median())

    return df


# shorten content_original
def shorten_content(df, is_title, is_every, is_similar, is_last, is_next, is_include, is_first, is_finally, merge_name):
    """
    :param df:
    :param is_title:  是否包含标题
    :param is_every:  是否包含每一个最相似行
    :param is_similar:  是否包含最相似行
    :param is_last:  是否包含最相似行的上一行
    :param is_next: 是否包含最相似行的下一行
    :param is_include: 是否包含出现在问题中的行+上一行+下一行
    :param is_first: 是否包含第一行
    :param is_finally: 是否包含最后一行
    :return: df
    """
    def match(title, content, question):
        result = []
        if is_title:
            result.append(title)

        content_list = content.split('。')
        temp = []
        for c in content_list:
            if c not in ['', ' ']:
                temp.append(c.strip())
        content_list = temp

        question_set = set(jieba.lcut(question))
        question_set = question_set - {'', ' '}
        scores = []
        for c in content_list:
            c_set = set(jieba.lcut(c))
            if c in question:
                scores.append(-1)
                continue
            score = len(c_set & question_set)
            scores.append(score)

        max_score = max(scores)
        for i in range(len(scores)):
            if scores[i] == max_score or (scores[i] < 0 and is_include):
                if i-1 >= 0 and is_last:
                    result.append(content_list[i-1])
                if is_similar:
                    result.append(content_list[i])
                if i+1 < len(content_list) and is_next:
                    result.append(content_list[i+1])
                if not is_every:
                    break
        if is_first:
            result.append(content_list[0])
        if is_finally:
            result.append(content_list[-1])

        # 过滤
        temp = []
        for r in result:
            if r not in temp:
                temp.append(r)
        result = temp

        return '。'.join(result)

    titles = df['title'].values
    contents = df['content'].values
    questions = df['question'].values

    merge = [match(t, c, q) for t, c, q in zip(titles, contents, questions)]
    df[merge_name] = merge

    # 评估数据集构建效果
    if 'answer' in df:
        answers = df['answer'].values
        is_in = [True if a in m else False for m, a in zip(merge, answers)]
        df[merge_name+'_in'] = is_in
        print('shorten content, accuracy: %.4f' % (sum(is_in)/len(df)))

    merge_len = [len(jieba.lcut(m)) for m in merge]
    df[merge_name+'_len'] = merge_len
    print('max length: %d' % max(merge_len))
    print('min length: %d' % min(merge_len))
    print('mean length:%d' % df[merge_name+'_len'].mean())

    temp = df[merge_name+'_len'].value_counts()[list(range(100000))]
    temp = temp[temp.notnull()].cumsum() / len(df)
    split_len = temp[temp > 0.98].index[0]
    print('split length(data>0.98): %d' % split_len)
    print('mean length(data>0.98): %d' % int(df[df[merge_name+'_len'] <= split_len][merge_name+'_len'].mean()))
    print('median length(data>0.98): %d\n' % int(df[df[merge_name+'_len'] <= split_len][merge_name+'_len'].median()))

    return df, split_len


# build answer_range
def build_answer_range(df):
    sys.setrecursionlimit(1000000)
    rouge = Rouge(metrics=['rouge-l'])

    def match(merge, answer, question):
        merge_list = jieba.lcut(merge)
        merge_len = len(merge_list)
        answer_list = jieba.lcut(answer)
        answer_len = len(answer_list)
        question_str = ' '.join(jieba.lcut(question))
        start = []
        end = []
        for i in range(0, merge_len-answer_len+1):
            if merge_list[i: i+answer_len] == answer_list:
                start.append(i)
                end.append(i+answer_len-1)
        if len(start) == 0:
            return -1, -1
        elif len(start) == 1:
            return start[0], end[0]
        else:
            scores = []
            # 前后扩展5个词
            for s, e in zip(start, end):
                s = max(s-5, 0)
                answer_can = ' '.join(merge_list[s: e+5])
                score = rouge.get_scores(answer_can, question_str, avg=True)['rouge-l']['r']
                scores.append(score)
            max_score = max(scores)
            start_tmp = []
            end_tmp = []
            for i in range(len(scores)):
                if scores[i] == max_score:
                    start_tmp.append(start[i])
                    end_tmp.append(end[i])
            start = start_tmp
            end = end_tmp
            if len(start) == 1:
                return start[0], end[0]
            else:
                return -2, -2

    merges = df[df['is_in']]['merge'].values
    answers = df[df['is_in']]['answer'].values
    questions = df[df['is_in']]['question'].values
    answer_range = [match(m, a, q) for m, a, q in zip(merges, answers, questions)]

    start, end = list(zip(*answer_range))
    df.loc[df['is_in'], 'answer_start'] = start
    df.loc[df['is_in'], 'answer_end'] = end

    merge_len = len(merges)
    right_all_len = (df['answer_end'] >= 0).sum()
    wrong_split_len = (df['answer_end'] == -1).sum()
    wrong_dup_len = (df['answer_end'] == -2).sum()
    print('answer generation accuracy(all): %.4f' % (right_all_len/merge_len))
    print('wrong split: %.4f' % (wrong_split_len/merge_len))
    print('answer duplicate: %.4f' % (wrong_dup_len/merge_len))

    return df


# build train, val, test dataset
def split_dataset(df):
    # deal data: 能找到答案
    all_data = len(df)
    print('all data size:%d' % all_data)
    # split train, val dataset
    train_df, val_df = model_selection.train_test_split(df, test_size=0.1, random_state=0)
    test_df = val_df.copy()

    # deal train, val data
    train_len = len(train_df)
    train_df = train_df[train_df['answer_start'] > -1]
    train_df = train_df[train_df['answer_end'] > -1]
    train_df = train_df[['question', 'merge', 'answer_start', 'answer_end']]
    print('train size:%d, shorten train size:%d' % (train_len, len(train_df)))

    # deal val data
    val_len = len(val_df)
    val_df = val_df[val_df['answer_start'] > -1]
    val_df = val_df[val_df['answer_end'] > -1]
    val_df = val_df[['question', 'merge', 'answer_start', 'answer_end']]
    print('val size:%d, shorten val size:%d' % (val_len, len(val_df)))

    # deal test data
    test_df = test_df

    return train_df, val_df, test_df


# collect data
# 1. title, content, question
# 2. split word and write to .txt
def collect_data(df):
    df = df[['title', 'content', 'question']]
    data = df.values.flatten().tolist()
    data = [' '.join(jieba.lcut(d)) for d in data]

    # write
    with open(config.collect_txt, 'w') as file:
        for d in data:
            file.writelines(d+'\n')


# generate vocab based on 'data_gen/collect_txt'
def gen_vocab():
    data_path = config.collect_txt
    lang = vocab.Vocab()
    with open(data_path, 'r') as file:
        for sentence in file.readlines():
            word_list = sentence.split()
            lang.add(word_list)
    lang.save(config.vocab_path)
    print('vocab length: %d' % len(lang.w2i))


# generate w2v based on 'data_gen/collect.txt'
def gen_w2v():
    data_file = config.collect_txt
    dim = config.w2i_size
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(
        sentences=LineSentence(data_file),
        size=dim,
        min_count=1,
        iter=5
    )
    lang = vocab.Vocab()
    lang.load(config.vocab_path)
    embedding = np.random.normal(size=(len(lang.w2i), model.wv.vector_size))
    for k, v in lang.w2i.items():
        if k in model.wv:
            embedding[v] = model.wv[k]
    np.save('data_gen/embedding_w2v_' + str(dim), embedding)


def gen_pre_file():
    if (os.path.isfile(config.collect_txt) is False) or (os.path.isfile(config.vocab_path) is False) or \
            (os.path.isfile(config.embedding_path) is False):
        # read .json
        df = organize_data(config.train_data)
        # 预处理数据
        df = deal_data(df)

    # 处理df， 生成collect.txt
    if os.path.isfile(config.collect_txt) is False:
        collect_data(df)
    # 生成词表
    if os.path.isfile(config.vocab_path) is False:
        gen_vocab()
    # 生成 embedding
    if os.path.isfile(config.embedding_path) is False:
        gen_w2v()


def gen_train_datafile():
    # read .json
    df = organize_data(config.train_data)
    # 预处理数据
    df = deal_data(df)
    # shorten content
    df = shorten_content_all(df, config.max_len)
    # answer_range
    df = build_answer_range(df)
    # split train, val
    df_train, df_val, df_test = split_dataset(df)
    # to .csv
    df_train.to_csv(config.train_df, index=False)
    df_val.to_csv(config.val_df, index=False)
    df_test.to_csv(config.test_df, index=False)


def gen_test_datafile():
    # read .json
    df = organize_data(config.test_data)
    # 预处理数据
    df = deal_data(df)
    # shorten content
    df = shorten_content_all(df, config.max_len)
    # to .csv
    df.to_csv(config.true_test_df, index=False)

if __name__ == '__main__':
    gen_train_datafile()





















