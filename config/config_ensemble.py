# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    model_bi_raf_1 = 'bi_daf_1'
    model_bi_raf_2 = 'bi_daf_1_mrt'

    model_r_net_1 = 'r_net_3_1'

    model_lst = [model_bi_raf_1, model_bi_raf_2]
    model_weight = [1, 1]
    model_weight = utils.softmax(model_weight)

    is_true_test = False

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
