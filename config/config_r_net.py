# coding = utf-8
# author = xy

from config import config_base


class Config(config_base.ConfigBase):
    model_name = 'r_net'
    model_save = model_name + '_1'
    is_bn = True
    epoch = 10
    mode = 'GRU'
    batch_size = 32
    test_batch_size = 16
    hidden_size = 75
    encoder_layer_num = 1
    dropout_p = 0.2
    val_every = 100

    embedding_type = 'expand'  # standard

    # 联合训练
    is_for_rouge = False
    if is_for_rouge:
        criterion = 'RougeLoss'
    lamda = 2  # 5

    # 测试
    model_test = 'r_net_1'
    is_true_test = False

config = Config()
