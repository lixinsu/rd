# coding = utf-8
# author = xy

from config import config_base
import utils


class Config(config_base.ConfigBase):
    ensemble_name = 'ensemble_1'

    model_bi_raf_1 = 'bi_daf_1'  # 0.9094

    model_m_reader_1 = 'm_reader_1'  # 0.9093

    model_m_reader_plus_1 = 'm_reader_plus_1'  # 0.9098
    model_m_reader_plus_1_mrt = 'm_reader_plus_1_mrt'  # 0.9128

    model_m_reader_plus_2 = 'm_reader_plus_2'  # 0.9113
    model_m_reader_plus_2_mrt = 'm_reader_plus_2_mrt'  # 0.9132

    model_m_reader_plus_3 = 'm_reader_plus_3'  # 0.9111

    model_lst = [model_m_reader_plus_1, model_m_reader_plus_2, model_m_reader_plus_3]
    model_weight = [0.9098, 0.9113, 0.9111]
    # model_weight = utils.softmax(model_weight)
    model_weight = utils.mean(model_weight)

    is_true_test = False

    if is_true_test:
        model_lst = [model + '_submission.pkl' for model in model_lst]
    else:
        model_lst = [model + '_val.pkl' for model in model_lst]


config = Config()
