# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'r_net'
    model_save = model_name + '_1'
    is_bn = False
    epoch = 10
    mode = 'GRU'
    batch_size = 32
    hidden_size = 75
    encoder_layer_num = 3
    dropout_p = 0.2
    val_every = 100

config = Config()
