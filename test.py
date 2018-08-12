# coding = utf-8
# author = xy

import os
import json
import pandas as pd
import torch
from my_metrics import blue
from my_metrics import rouge
import loader
import utils
import preprocess_data
from config import config_base
from config import config_merge
from modules import match_lstm


# config
config = config_base.config


def test():
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
        test_data = loader.load_data(config.test_df, config.merge_name, lang)
    else:
        test_data = loader.load_data(config.true_test_df, config.merge_name, lang)

    # build test dataloader
    test_loader = loader.build_loader(
        dataset=test_data[:2],
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )

    # model initial
    param = {
        'embedding': embedding_np,
        'embedding_type': config.embedding_type,
        'encoder_mode': config.encoder_mode,
        'hidden_size': config.hidden_size,
        'dropout_p': config.dropout_p,
        'encoder_dropout_p': config.encoder_dropout_p,
        'encoder_bidirectional': config.encoder_bidirectional,
        'encoder_layer_num': config.encoder_layer_num
    }
    model = eval(config.model_name).Model(param)
    model = model.cuda()

    # load model param, and training state
    model_path = os.path.join('model', config.model_save)
    state = torch.load(model_path)
    model.load_state_dict(state['best_model_state'])

    best_loss = state['best_loss']
    best_epoch = state['best_epoch']
    best_step = state['best_step']
    time_use = state['time']
    print('best_epoch:%2d, best_step:%5d, best_loss:%.4f, best_time:%d' % (best_epoch, best_step, best_loss, time_use))

    # gen result
    result = []
    result_start = []
    result_end = []
    model.eval()
    for batch in test_loader:
        # cuda, cut
        batch = utils.deal_batch(batch)
        outputs = model(batch)

        _, start = torch.topk(outputs[0], k=1, dim=1)
        start = start.reshape(-1).cpu().numpy().tolist()
        _, end = torch.topk(outputs[1], k=1, dim=1)
        end = end.reshape(-1).cpu().numpy().tolist()

        content = batch[0].cpu().numpy()
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
        rouge_score = rouge.RougeL()
        for a, r in zip(answer_true, result):
            blue_score.add_inst(r, a)
            rouge_score.add_inst(r, a)
        print('rouge_L score: %.4f, blue score:%.4f' % (rouge_score.get_score(), blue_score.get_score()))

    # to .csv
    if True:
        df[config.merge_name + '_answer_pred'] = result
        df[config.merge_name + '_answer_start_pred'] = result_start
        df[config.merge_name + '_answer_end_pred'] = result_end

        if config.merge_name+'_answer_start' in df:
            df = df[['article_id', 'title', 'content', config.merge_name, 'question', 'answer', config.merge_name+'_answer_pred',
                     config.merge_name+'_answer_start', config.merge_name+'_answer_end',
                     config.merge_name+'_answer_start_pred', config.merge_name+'_answer_end_pred']]
            csv_path = os.path.join('result', config.model_save+'_val.csv')

        else:
            df = df[['article_id', 'title', 'content', config.merge_name, 'question', config.merge_name+'_answer_pred',
                     config.merge_name+'_answer_start_pred', config.merge_name+'_answer_end_pred']]
            csv_path = os.path.join('result', config.model_save+'_submission.csv')

        df.to_csv(csv_path, index=False)


def test_ensemble():
    pass


if __name__ == '__main__':
    test()
