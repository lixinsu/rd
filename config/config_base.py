# coding = utf-8
# author = xy


class ConfigBase:
    train_data = 'data/first_question.json'
    test_data = 'data/first_question.json'

    merge_name = 'data2'
    train_df = 'data_gen/' + merge_name + '_train.csv'
    val_df = 'data_gen/' + merge_name + '_val.csv'
    test_df = 'data_gen/' + merge_name + '_test.csv'

    fake_test_df = 'data_gen/' + merge_name + '_true_test.csv'
    true_test_df = fake_test_df

    is_true_test = False
    submission_file = 'submission/test.json'

    collect_txt = 'data_gen/collect.txt'
    vocab_path = 'data_gen/vocab.pkl'

    w2i_size = 256
    embedding_path = 'data_gen/embedding_w2v_' + str(w2i_size) + '.npy'
    embedding_is_training = True

    model_name = ['r_net', 'match_lstm'][0]
    model_save = merge_name + '_' + model_name + '_2'
    criterion = 'MyNLLLoss'
    embedding_type = 'standard'
    is_bn = False
    lr = 1e-4
    epoch = 10
    mode = 'GRU'
    batch_size = 32
    hidden_size = 75
    encoder_layer_num = 2
    encoder_bidirectional = True
    encoder_dropout_p = 0.1
    dropout_p = 0.2
    max_grad = 5
    val_every = 100

config = ConfigBase()
