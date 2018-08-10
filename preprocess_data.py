# coding = utf-8
# author = xy

import json
import os
import pandas as pd
import numpy as np
import copy
import jieba
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
        answers = [answer[:-1] if answer[-1] in drop_list else answer for answer in answers]
        df.loc[df['answer'] != '', 'answer'] = answers

    return df


# shorten content
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
def build_answer_range(df, merge_name):
    def match(merge, answer):
        merge_list = jieba.lcut(merge)
        merge_len = len(merge_list)
        answer_list = jieba.lcut(answer)
        answer_len = len(answer_list)
        start = -1
        end = -1
        for i in range(0, merge_len-answer_len+1):
            if merge_list[i: i+answer_len] == answer_list:
                start = i
                end = i+answer_len-1
                break

        return start, end

    merges = df[df[merge_name+'_in']][merge_name].values
    answers = df[df[merge_name+'_in']]['answer'].values
    answer_range = [match(m, a) for m, a in zip(merges, answers)]

    start, end = list(zip(*answer_range))
    df.loc[df[merge_name+'_in'], merge_name+'_answer_start'] = start
    df.loc[df[merge_name+'_in'], merge_name+'_answer_end'] = end

    merge_len = len(merges)
    right_len = (df[merge_name+'_answer_start'] > -1).sum()
    print('answer generation accuracy: %.4f\n' % (right_len/merge_len))

    return df


# build train, val, test dataset
def split_dataset(df, split_len, merge_name):
    # deal data: 能找到答案， 长度限制（0.98）
    all_data = len(df)
    print('all data size:%d' % all_data)
    # split train, val dataset
    train_df, val_df = model_selection.train_test_split(df, test_size=0.1, random_state=0)
    test_df = val_df.copy()

    # deal train, val data
    train_len = len(train_df)
    train_df = train_df[train_df[merge_name+'_answer_start'] > -1]
    train_df = train_df[train_df[merge_name+'_answer_end'] > -1]
    train_df = train_df[train_df[merge_name+'_len'] <= split_len]
    train_df = train_df[['question', merge_name, merge_name+'_answer_start', merge_name+'_answer_end']]
    print('train size:%d, shorten train size:%d' % (train_len, len(train_df)))

    # deal val data
    val_len = len(val_df)
    val_df = val_df[val_df[merge_name+'_answer_start'] > -1]
    val_df = val_df[val_df[merge_name+'_answer_end'] > -1]
    val_df = val_df[val_df[merge_name+'_len'] <= split_len]
    val_df = val_df[['question', merge_name, merge_name+'_answer_start', merge_name+'_answer_end']]
    print('val size:%d, shorten val size:%d' % (val_len, len(val_df)))

    # deal test data
    merge_len = test_df[merge_name+'_len'].values
    merge = test_df[merge_name].values
    merge_fix = []
    c = 0
    for l, m in zip(merge_len, merge):
        if l > split_len:
            c += 1
            m = jieba.lcut(m)[:split_len]
            m = ''.join(m)
        merge_fix.append(m)
    print('test size:%d, fix size:%d' % (len(test_df), c))
    test_df[merge_name] = merge_fix

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
    df, split_data = shorten_content(
        df=df,
        is_title=True,
        is_every=False,
        is_similar=True,
        is_last=False,
        is_next=True,
        is_first=False,
        is_finally=False,
        is_include=False,
        merge_name=config.merge_name
    )
    # answer_range
    df = build_answer_range(df, config.merge_name)
    # split train, val
    df_train, df_val, df_test = split_dataset(df, split_data, config.merge_name)
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
    df, split_data = shorten_content(
        df=df,
        is_title=True,
        is_every=False,
        is_similar=True,
        is_last=False,
        is_next=True,
        is_first=False,
        is_finally=False,
        is_include=False,
        merge_name=config.merge_name
    )

    # deal test data
    merge_len = df[config.merge_name+'_len'].values
    merge = df[config.merge_name].values
    merge_fix = []
    c = 0
    for l, m in zip(merge_len, merge):
        if l > split_data:
            c += 1
            m = jieba.lcut(m)[:split_data]
            m = ''.join(m)
        merge_fix.append(m)
    print('test size:%d, fix size:%d' % (len(df), c))
    df[config.merge_name] = merge_fix

    # to .csv
    df.to_csv(config.true_test_df, index=False)

if __name__ == '__main__':
    pass
