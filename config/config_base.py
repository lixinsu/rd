# coding = utf-8
# author = xy


class ConfigBase:
    train_data = 'data/first_question.json'
    test_data = None

    train_df = 'data_gen/train_df.csv'
    test_df = 'data_gen/test_df.csv'

    train_df_done = 'data_gen/train_df_done.csv'
    val_df_done = 'data_gen/val_df_done.csv'
    test_df_done = 'data_gen/test_df_done.csv'

    collect_txt = 'data_gen/collect.txt'

    vocab_size = -1
    vocab_path = 'data_gen/vocab_all.pkl'

    w2i_size = 256
    embedding_path = 'data_gen/embedding_w2v_' + str(w2i_size) + '.txt'

    model_name = 'match_lstm'
    model_save = 'match_lstm'
    criterion = 'MyNLLLoss'
    embedding_type = 'standard'
    lr = 1e-4
    epoch = 10
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
