# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'match_lstm_plus'
    model_save = model_name + '_2'
    is_bn = True
    epoch = 12
    mode = 'GRU'
    batch_size = 32
    hidden_size = 150
    encoder_layer_num = 1
    dropout_p = 0.3
    val_every = 100
    val_mean = False

    # 联合训练
    is_for_rouge = False
    if is_for_rouge:
        criterion = 'RougeLoss'
    lamda = 2  # 5

    # 测试
    model_test = 'match_lstm_plus_2'
    is_true_test = False

config = Config()
