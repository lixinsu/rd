# coding = utf-8
# author = xy

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import loader
import torch
from torch import optim
from torch import nn
from config import config_base
from config import config_merge
import preprocess_data
import utils
from modules.layers.loss import MyNLLLoss
from modules import match_lstm


# config
config = config_base.config


def train():
    time_start = time.time()
    # prepare: collect, vocab, embedding
    preprocess_data.gen_pre_file()
    # load vocab
    lang = loader.load_vocab(config.vocab_path)
    # load w2v
    embedding_np = loader.load_w2v(config.embedding_path)

    # prepare: train_df, val_df
    if (os.path.isfile(config.train_df) is False) or (os.path.isfile(config.val_df) is False) or \
            (os.path.isfile(config.test_df) is False):
        print('gen train_df.csv, val_df.csv, test_df.csv')
        time0 = time.time()
        preprocess_data.gen_train_datafile()
        print('gen train_df.csv, val_df.csv, test_df.csv. time:%d' % (time.time()-time0))
    # load data: merge, question, answer_start, answer_end
    print('load data...')
    time0 = time.time()
    train_data = loader.load_data(config.train_df, config.merge_name, lang)
    val_data = loader.load_data(config.val_df, config.merge_name, lang)
    print('load data finished, time:%d' % (time.time()-time0))

    # build train, val dataloader
    train_loader = loader.build_loader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = loader.build_loader(
        dataset=val_data,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True
    )

    # model:
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

    # loss
    criterion = eval(config.criterion)()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # load model param, optimizer param, train param
    model_path = os.path.join('model', config.model_save)
    if os.path.isfile(model_path):
        state = torch.load(model_path)
        model.load_state_dict(state['cur_model_state'])
        optimizer.load_state_dict(state['cur_opt_state'])
        epoch_list = range(state['cur_epoch']+1, state['cur_epoch']+1+config.epoch)
        train_loss_list = state['train_loss']
        val_loss_list = state['val_loss']
        steps = state['steps']
        time_use = state['time']
    else:
        state = None
        epoch_list = range(config.epoch)
        train_loss_list = []
        val_loss_list = []
        steps = []
        time_use = 0

    # train
    model_param_num = 0
    for param in model.parameters():
        model_param_num += param.nelement()
    print('starting training....')
    if state is None:
        print('start_epoch:0, end_epoch:%d, num_params:%d, num_params_except_embedding:%d' %
              (config.epoch-1, model_param_num, model_param_num-embedding_np.shape[0]*embedding_np.shape[1]))
    else:
        print('start_epoch:%d, end_epoch:%d, num_params:%d, num_params_except_embedding:%d' %
              (state['cur_epoch']+1, state['cur_epoch']+config.epoch, model_param_num,
               model_param_num-embedding_np.shape[0]*embedding_np.shape[1]))

    plt.ion()
    train_loss = 0
    train_c = 0
    for e in epoch_list:
        for i, batch in enumerate(train_loader):
            # cut, cuda
            batch = utils.deal_batch(batch)

            model.train()
            optimizer.zero_grad()
            outputs = model(batch)
            loss_value = criterion(outputs, batch)
            loss_value.backward()

            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad)
            optimizer.step()

            train_loss += loss_value.item()
            train_c += 1

            if train_c % config.val_every == 0:
                val_loss = 0
                val_c = 0
                with torch.no_grad():
                    model.eval()
                    for val_batch in val_loader:
                        # cut, cuda
                        val_batch = utils.deal_batch(val_batch)

                        outputs = model(val_batch)
                        loss_value = criterion(outputs, val_batch)

                        val_loss += loss_value.item()
                        val_c += 1

                train_loss_list.append(train_loss/train_c)
                val_loss_list.append(val_loss/val_c)
                steps.append(config.val_every)

                print('training, epochs:%2d, steps:%5d, train_loss:%.4f, val_loss:%.4f, time:%4ds' %
                      (e, sum(steps), train_loss/train_c, val_loss/val_c, time.time()-time_start+time_use))

                train_loss = 0
                train_c = 0

                # draw
                plt.cla()
                x = np.cumsum(steps)
                plt.plot(
                    x,
                    train_loss_list,
                    color='r',
                    label='train'
                )
                plt.plot(
                    x,
                    val_loss_list,
                    color='b',
                    label='val'
                )
                plt.xlabel('steps')
                plt.ylabel('loss')
                plt.legend()
                plt.pause(0.0000001)
                fig_path = os.path.join('model', ''.join(list(config.model_save))+'.png')
                plt.savefig(fig_path)
                plt.show()

                # save model
                if os.path.isfile(model_path):
                    state = torch.load(model_path)
                else:
                    state = {}

                if state == {} or state['best_loss'] > (val_loss/val_c):
                    state['best_model_state'] = model.state_dict()
                    state['best_loss'] = val_loss/val_c
                    state['best_epoch'] = e
                    state['best_step'] = sum(steps)
                    state['best_time'] = time_use + time.time() - time_start

                state['cur_model_state'] = model.state_dict()
                state['cur_opt_state'] = optimizer.state_dict()
                state['cur_epoch'] = e
                state['train_loss'] = train_loss_list
                state['val_loss'] = val_loss_list
                state['steps'] = steps
                state['time'] = time_use + time.time() - time_start

                torch.save(state, model_path)


if __name__ == '__main__':
    train()


