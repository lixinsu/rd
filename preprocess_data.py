# coding = utf-8
# author = xy

import json
import pandas as pd
import copy
import jieba
import thulac
import re
import logging
import gensim
from gensim.models.word2vec import LineSentence
from sklearn import model_selection
from config import config_base
import vocab
import utils


config = config_base.config


# 1. convert .json to .pandas
# 2. drop unrelated column
def build_data(file_in, file_out):

    df = organize_data(file_in)
    df = shorten_content(df)


    # df.to_csv(file_out, index=False)
    return df


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
    df.loc[:, 'title'] = deal(df['article_title'].values)
    df.loc[:, 'content'] = deal(df['article_content'].values)
    df.loc[:, 'question'] = deal(df['article_question'].values)
    df.loc[:, 'answer'] = deal(df['article_answer'].values)

    # 3
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

        question_set = set(jieba.cut(question))
        question_set = question_set - {'', ' '}
        scores = []
        for c in content_list:
            c_set = set(jieba.cut(c))
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
    answers = df['answer'].values

    merge = [match(t, c, q) for t, c, q in zip(titles, contents, questions)]
    df[merge_name] = merge

    # 评估数据集构建效果
    is_in = [True if a in m else False for m, a in zip(merge, answers)]
    df[merge_name+'_in'] = is_in
    print('accuracy: %.4f' % (sum(is_in)/len(df)))

    merge_len = [len(list(jieba.cut(m))) for m in merge]
    df[merge_name+'_len'] = merge_len
    print('max length: %d' % max(merge_len))
    print('min length: %d' % min(merge_len))
    print('mean length:%d' % df[merge_name+'_len'].mean())

    temp = df[merge_name+'_len'].value_counts()[list(range(100000))]
    temp = temp[temp.notnull()].cumsum() / len(df)
    split_len = temp[temp > 0.98].index[0]
    print('split length(data>0.98): %d' % split_len)
    print('mean length(data>0.98): %d' % int(df[df[merge_name+'_len'] <= split_len][merge_name+'_len'].mean()))
    print('mean length(data>0.98): %d' % int(df[df[merge_name+'_len'] <= split_len][merge_name+'_len'].median()))

    return df


# build answer_range
def build_answer_range(df, merge_name):
    def match(merge, answer):
        merge_list = jieba.lcut(merge)
        answer_list = jieba.lcut(answer)

        point = merge.find(answer)
        temp = merge[: point]
        start = len(list(jieba.cut(temp)))
        answer_len = len(list(jieba.cut(answer)))
        end = start + answer_len - 1

        flag = True
        while flag:
            temp = ''.join(merge_list[start: end+1])
            if answer in temp:
                flag = False
            else:
                start = start - 1 if start > 0 else start
                end = end + 1

        flag = True
        while flag:
            temp_left = ''.join(merge_list[start+1: end+1])
            temp_right = ''.join(merge_list[start: end])

            if answer not in temp_left and answer not in temp_right:
                flag = False
            elif answer in temp_left:
                start = start + 1
            elif answer in temp_right:
                end = end - 1

        return start, end

    merges = df[df[merge_name+'_in']][merge_name].values
    answers = df[df[merge_name+'_in']]['answer'].values
    answer_range = [match(m, a) for m, a in zip(merges, answers)]

    start, end = list(zip(*answer_range))
    df.loc[df[merge_name+'_in'], merge_name+'_answer_start'] = start
    df.loc[df[merge_name+'_in'], merge_name+'_answer_end'] = end

    merges = [jieba.lcut(m) for m in merges]
    answer_gen = [''.join(m[s: e+1]) for m, s, e, in zip(merges, start, end)]
    accuracy = [True if fake == real else False for fake, real in zip(answer_gen, answers)]
    print('answer generation accuracy %.4f' % (sum(accuracy)/len(answer_gen)))

    return df











# collect data
# 1. title, content, question, answer
# 2. split word
def collect_data():

    train_data_path = config.train_df
    test_data_path = config.test_df

    df_train = pd.read_csv(train_data_path)[['article_title', 'article_content', 'question', 'answer']]
    data = df_train.values.flatten().tolist()
    df_test = pd.read_csv(test_data_path)[['article_title', 'article_content', 'question']]
    data = data + df_test.values.flatten().tolist()
    data = [' '.join(list(jieba.cut(d))) for d in data]
    data = [re.sub(r'\s+', ' ', d) for d in data]

    # write
    with open(config.collect_txt, 'w') as file:
        for d in data:
            file.writelines(d+'\n')


# generate vocab based on 'data_gen/collect_txt'
def gen_vocab(size=config.vocab_size):
    data_path = config.collect_txt
    lang = vocab.Vocab()
    with open(data_path, 'r') as file:
        for sentence in file.readlines():
            word_list = sentence.split()
            lang.add(word_list)
    if size != -1:
        # deal
        lang.save('data_gen/vocab_' + str(size) + '.pkl')
    else:
        lang.save('data_gen/vocab_all.pkl')

    print('vocab length: %d' % len(lang.w2i))


# generate w2v based on 'data_gen/collect.txt'
def gen_w2v(dim=config.w2i_size):
    data_file = config.collect_txt
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(
        sentences=LineSentence(data_file),
        size=dim,
        min_count=1,
        iter=5
    )
    model.wv.save_word2vec_format('data_gen/embedding_w2v_' + str(dim) + '.txt')


# 待定
def deal_dat():
    """
    1. split word
    2. index
    3. location answer
    4. split train, val dataset
    5. save
    """
    data_path = ['data_gen/train_df.csv', 'data_gen/test_df.csv']
    lang = vocab.Vocab()
    lang.load(config.vocab_path)
    for path in data_path:
        df = pd.read_csv(path)

        # 1. split word
        content = list(map(lambda item: list(jieba.cut(item)), df['article_content'].values))
        question = list(map(lambda item: list(jieba.cut(item)), df['question'].values))

        # 2. index
        content = [lang.words2indexes(c) for c in content]
        question = [lang.words2indexes(q) for q in question]

        # 3. location answer
        answer_start = 0
        answer_end = 0

        df['content_index'] = content
        df['question_index'] = question
        if 'answer' in df.columns:
            df['answer_start'] = answer_start
            df['answer_end'] = answer_end

        # 4. split train, val dataset, save
        if 'answer' in df.columns:
            df_train, df_val = model_selection.train_test_split(df, test_size=0.1, random_state=0)
            df_train.to_csv(config.train_df_done, index=False)
            df_val.to_csv(config.val_df_done)
        else:
            df.to_csv(config.test_df_done)

if __name__ == '__main__':
    organize_data()




















