# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'r_net'
    model_save = model_name + '_2'
    is_bn = True
    epoch = 10
    mode = 'GRU'
    batch_size = 32
    hidden_size = 75
    encoder_layer_num = 2
    dropout_p = 0.2
    val_every = 100

    is_true_test = False

config = Config()
