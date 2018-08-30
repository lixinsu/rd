# coding = utf-8
# author = xy

import os
import time
import json
import pickle
import pandas as pd
import torch
from my_metrics import blue
from my_metrics import rouge_test
import loader
import utils
import preprocess_data
from config import config_base
from config import config_r_net
from config import config_match_lstm
from config import config_bi_daf
from config import config_qa_net
from config import config_ensemble
from modules import match_lstm
from modules import r_net
from modules import bi_daf
from modules import qa_net

# config = config_match_lstm.config
# config = config_r_net.config
# config = config_bi_daf.config
# config = config_ensemble.config
config = config_qa_net.config


def test():
    time0 = time.time()
    # load vocab
    lang = loader.load_vocab(config.vocab_path)
    # load w2v
    embedding_np = loader.load_w2v(config.embedding_path)

    # prepare: test_df
    if config.is_true_test and (os.path.isfile(config.true_test_df) is False):
        preprocess_data.gen_test_datafile()

    if (config.is_true_test is False) and (os.path.isfile(config.test_df) is False):
        preprocess_data.gen_train_datafile()

    # load data
    if config.is_true_test is False:
        if os.path.isfile(config.test_pkl):
            with open(config.test_pkl, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = loader.load_data(config.test_df, lang)
            with open(config.test_pkl, 'wb') as file:
                pickle.dump(test_data, file)

    else:
        if os.path.isfile(config.true_test_pkl):
            with open(config.true_test_pkl, 'rb') as file:
                test_data = pickle.load(file)
        else:
            test_data = loader.load_data(config.true_test_df, lang)
            with open(config.true_test_pkl, 'wb') as file:
                pickle.dump(test_data, file)

    # build test dataloader
    test_loader = loader.build_loader(
        dataset=test_data[:6],
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False
    )

    # model initial
    param = {
        'embedding': embedding_np,
        'embedding_type': config.embedding_type,
        'embedding_is_training': config.embedding_is_training,
        'mode': config.mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': config.encoder_dropout_p,
        'encoder_bidirectional': config.encoder_bidirectional,
        'encoder_layer_num': config.encoder_layer_num,
        'is_bn': config.is_bn
    }

    model = eval(config.model_name).Model(param)
    model = model.cuda()

    # load model param, and training state
    model_path = os.path.join('model', config.model_test)
    print('load model, ', model_path)
    state = torch.load(model_path)
    model.load_state_dict(state['best_model_state'])

    best_loss = state['best_loss']
    best_epoch = state['best_epoch']
    best_step = state['best_step']
    best_time = state['best_time']
    use_time = state['time']
    print('best_epoch:%2d, best_step:%5d, best_loss:%.4f, best_time:%d, use_time:%d' %
          (best_epoch, best_step, best_loss, best_time, use_time))

    # gen result
    result = []
    result_start = []
    result_end = []
    result_ans_range = []
    model.eval()
    for batch in test_loader:
        # cuda, cut
        batch = utils.deal_batch(batch)
        outputs = model(batch)
        start, end = utils.answer_search(outputs)

        start = start.reshape(-1).cpu().numpy().tolist()
        end = end.reshape(-1).cpu().numpy().tolist()

        content = batch[0].cpu().numpy()
        result_batch = [c[s: e+1] for s, e, c in zip(start, end, content)]

        result = result + result_batch
        result_start = result_start + start
        result_end = result_end + end

        result_ans_range.append(outputs.data.cpu())

    result = [lang.indexes2words(r) for r in result]
    result = [''.join(r) for r in result]

    if config.is_true_test:
        df = pd.read_csv(config.true_test_df)
    else:
        df = pd.read_csv(config.test_df)

    # gen a submission
    if config.is_true_test:
        articled_ids = df['article_id'].astype(str).values.tolist()
        question_ids = df['question_id'].values
        submission = []
        temp_a_id = articled_ids[0]
        temp_qa = []
        for a_id, q_id, a in zip(articled_ids, question_ids, result):
            if a_id == temp_a_id:
                sub = {'questions_id': q_id, 'answer': a}
                temp_qa.append(sub)
            else:
                submission.append({'article_id': temp_a_id, 'questions': temp_qa})
                temp_a_id = a_id
                temp_qa = [{'questions_id': q_id, 'answer': a}]
        submission.append({'article_id': temp_a_id, 'questions': temp_qa})

        submission_article = [s['article_id'] for s in submission]
        submission_questions = [s['questions'] for s in submission]
        submission_dict = dict(zip(submission_article, submission_questions))

        with open(config.test_data, 'r') as file:
            all_data = json.load(file)
        all_article = [d['article_id'] for d in all_data]

        submission = []
        for a_id in all_article:
            if a_id in submission_dict:
                submission.append({'article_id': a_id, 'questions': submission_dict[a_id]})
            else:
                submission.append({'article_id': a_id, 'questions': []})

        with open(config.submission_file, mode='w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False)

    # my_metrics
    if config.is_true_test is False:
        answer_true = df['answer'].values
        assert len(result) == len(answer_true)
        blue_score = blue.Bleu()
        rouge_score = rouge_test.RougeL()
        for a, r in zip(answer_true, result):
            blue_score.add_inst(r, a)
            rouge_score.add_inst(r, a)
        print('rouge_L score: %.4f, blue score:%.4f' % (rouge_score.get_score(), blue_score.get_score()))

    # to .csv
    if True:
        df['answer_pred'] = result
        df['answer_start_pred'] = result_start
        df['answer_end_pred'] = result_end

        if 'answer_start' in df:
            df = df[['article_id', 'title', 'content', 'merge', 'question', 'answer', 'answer_pred',
                     'answer_start', 'answer_end', 'answer_start_pred', 'answer_end_pred']]
            csv_path = os.path.join('result', config.model_test+'_val.csv')

        else:
            df = df[['article_id', 'title', 'content', 'merge', 'question', 'answer_pred',
                     'answer_start_pred', 'answer_end_pred']]
            csv_path = os.path.join('result', config.model_test+'_submission.csv')

        df.to_csv(csv_path, index=False)

    # save result_ans_range
    if config.is_true_test:
        save_path = os.path.join('result/ans_range', config.model_test+'_submission.pkl')
    else:
        save_path = os.path.join('result/ans_range', config.model_test+'_val.pkl')
    torch.save(result_ans_range, save_path)
    print('time:%d' % (time.time()-time0))


def test_ensemble():
    time0 = time.time()
    # 加权求和
    model_lst = config.model_lst
    model_weight = config.model_weight
    ans_range_list = []
    for model_result in model_lst:
        result_path = os.path.join('result/ans_range', model_result)
        ans_range = torch.load(result_path)
        ans_range_list.append(ans_range)

    model_num = len(model_lst)
    batchs = len(ans_range_list[0])
    ans_range_ensemble = []
    for i in range(batchs):
        ans_range = ans_range_list[0][i].new_zeros(ans_range_list[0][i].size())
        for j in range(model_num):
            ans_range += ans_range_list[j][i] * model_weight[j]
        ans_range_ensemble.append(ans_range)

    # load vocab
    lang = loader.load_vocab(config.vocab_path)

    # prepare: test_df
    if config.is_true_test and (os.path.isfile(config.true_test_df) is False):
        preprocess_data.gen_test_datafile()

    if (config.is_true_test is False) and (os.path.isfile(config.test_df) is False):
        preprocess_data.gen_train_datafile()

    # load data
    if config.is_true_test is False:
        test_data = loader.load_data(config.test_df, lang)
    else:
        test_data = loader.load_data(config.true_test_df, lang)

    # build test dataloader
    test_loader = loader.build_loader(
        dataset=test_data[:2],
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False
    )

    # 建立content
    contents = []
    for batch in test_loader:
        contents.append(batch[0].numpy())

    # 生成结果
    result = []
    result_start = []
    result_end = []
    for ans_range, content in zip(ans_range_ensemble, contents):
        start, end = utils.answer_search(ans_range)

        start = start.reshape(-1).cpu().numpy().tolist()
        end = end.reshape(-1).cpu().numpy().tolist()

        result_batch = [c[s: e+1] for s, e, c in zip(start, end, content)]

        result = result + result_batch
        result_start = result_start + start
        result_end = result_end + end

    result = [lang.indexes2words(r) for r in result]
    result = [''.join(r) for r in result]

    if config.is_true_test:
        df = pd.read_csv(config.true_test_df)
    else:
        df = pd.read_csv(config.test_df)

    # gen a submission
    if config.is_true_test:
        articled_ids = df['article_id'].astype(str).values.tolist()
        question_ids = df['question_id'].values
        submission = []
        temp_a_id = articled_ids[0]
        temp_qa = []
        for a_id, q_id, a in zip(articled_ids, question_ids, result):
            if a_id == temp_a_id:
                sub = {'questions_id': q_id, 'answer': a}
                temp_qa.append(sub)
            else:
                submission.append({'article_id': temp_a_id, 'questions': temp_qa})
                temp_a_id = a_id
                temp_qa = [{'questions_id': q_id, 'answer': a}]
        submission.append({'article_id': temp_a_id, 'questions': temp_qa})

        submission_article = [s['article_id'] for s in submission]
        submission_questions = [s['questions'] for s in submission]
        submission_dict = dict(zip(submission_article, submission_questions))

        with open(config.test_data, 'r') as file:
            all_data = json.load(file)
        all_article = [d['article_id'] for d in all_data]

        submission = []
        for a_id in all_article:
            if a_id in submission_dict:
                submission.append({'article_id': a_id, 'questions': submission_dict[a_id]})
            else:
                submission.append({'article_id': a_id, 'questions': []})

        with open(config.submission_file, mode='w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False)

    # my_metrics
    if config.is_true_test is False:
        answer_true = df['answer'].values
        assert len(result) == len(answer_true)
        blue_score = blue.Bleu()
        rouge_score = rouge_test.RougeL()
        for a, r in zip(answer_true, result):
            blue_score.add_inst(r, a)
            rouge_score.add_inst(r, a)
        print('rouge_L score: %.4f, blue score:%.4f' % (rouge_score.get_score(), blue_score.get_score()))

    # to .csv
    if True:
        df['answer_pred'] = result
        df['answer_start_pred'] = result_start
        df['answer_end_pred'] = result_end

        if 'answer_start' in df:
            df = df[['article_id', 'title', 'content', 'merge', 'question', 'answer', 'answer_pred',
                     'answer_start', 'answer_end', 'answer_start_pred', 'answer_end_pred']]
            csv_path = os.path.join('result', 'ensemble_val.csv')

        else:
            df = df[['article_id', 'title', 'content', 'merge', 'question', 'answer_pred',
                     'answer_start_pred', 'answer_end_pred']]
            csv_path = os.path.join('result', 'ensemble_submission.csv')

        df.to_csv(csv_path, index=False)

    print('time:%d' % (time.time()-time0))


if __name__ == '__main__':
    if config == config_ensemble.config:
        print('ensemble...')
        test_ensemble()
    else:
        print('single model...')
        test()
