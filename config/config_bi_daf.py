# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'bi_daf'
    model_save = model_name + '_1'
    is_bn = True
    epoch = 12
    mode = 'LSTM'
    batch_size = 32
    hidden_size = 100
    encoder_layer_num = 1
    dropout_p = 0.2
    val_every = 100
    val_mean = False  # 这个指标用来衡量，是否是每隔固定次数验证一次

    # 联合训练
    is_for_rouge = False
    if is_for_rouge:
        criterion = 'RougeLoss'
        lamda = 5

    # 测试
    model_test = 'bi_daf_1'
    is_true_test = True

config = Config()
