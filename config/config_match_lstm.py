# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'match_lstm'
    model_save = model_name + '_1'
    is_bn = True
    epoch = 10
    mode = 'LSTM'
    batch_size = 32
    hidden_size = 150
    encoder_layer_num = 2
    dropout_p = 0.4
    val_every = 100

    is_true_test = False

config = Config()
