# coding = utf-8
# author = xy


class ConfigBase:
    train_data = 'data/first_question.json'
    test_data = 'data/first_question.json'

    max_len = 500
    train_df = 'data_gen/merge_' + str(max_len) + '_train.csv'
    train_pkl = train_df[: -4] + '.pkl'  # index tag.....
    val_df = 'data_gen/merge_' + str(max_len) + '_val.csv'
    val_pkl = val_df[: -4] + '.pkl'
    test_df = 'data_gen/merge_' + str(max_len) + '_test.csv'
    test_pkl = test_df[: -4] + '.pkl'

    fake_test_df = 'data_gen/merge_' + str(max_len) + '_trueTest.csv'
    true_test_df = fake_test_df
    true_test_pkl = true_test_df[: -4] + '.pkl'

    is_true_test = False
    submission_file = 'submission/test.json'

    collect_txt = 'data_gen/collect.txt'
    vocab_path = 'data_gen/vocab.pkl'
    tag2index_path = 'data_gen/tag2index.pkl'
    word2tag_path = 'data_gen/word2tag.pkl'

    w2i_size = 256
    embedding_path = 'data_gen/embedding_w2v_' + str(w2i_size) + '.npy'
    embedding_is_training = True

    criterion = 'MyNLLLoss'
    embedding_type = 'expand'
    is_bn = True
    lr = 1e-4
    epoch = 10
    mode = 'GRU'
    batch_size = 32
    test_batch_size = 64
    hidden_size = 75
    encoder_layer_num = 2
    encoder_bidirectional = True
    encoder_dropout_p = 0.1
    dropout_p = 0.2
    max_grad = 5
    val_every = 100

config = ConfigBase()
