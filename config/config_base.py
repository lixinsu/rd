# coding = utf-8
# author = xy


class ConfigBase:
    train_data = 'data/first_question.json'
    test_data = 'data/first_question.json'

    merge_name = 'data2'
    train_df = 'data_gen/train_' + merge_name + '.csv'
    val_df = 'data_gen/val_' + merge_name + '.csv'
    test_df = 'data_gen/test_' + merge_name + '.csv'

    fake_test_df = 'data_gen/true_test_' + merge_name + '.csv'
    true_test_df = fake_test_df

    is_true_test = True
    submission_file = 'submission/test.json'

    collect_txt = 'data_gen/collect.txt'
    vocab_path = 'data_gen/vocab.pkl'

    w2i_size = 256
    embedding_path = 'data_gen/embedding_w2v_' + str(w2i_size) + '.npy'

    model_name = 'match_lstm'
    model_save = merge_name + '_' + model_name + '_1'
    criterion = 'MyNLLLoss'
    embedding_type = 'standard'
    lr = 1e-4
    epoch = 20
    encoder_mode = 'LSTM'
    batch_size = 32
    hidden_size = 256
    encoder_layer_num = 2
    encoder_bidirectional = False
    encoder_dropout_p = 0.1
    dropout_p = 0.4
    max_grad = 5
    val_every = 100

config = ConfigBase()
