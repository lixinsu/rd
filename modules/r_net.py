# coding = utf-8
# author = xy

from torch import nn
from modules.layers import embedding
from modules.layers import encoder
from modules.layers import match_rnn
from modules.layers import self_match_attention
from modules.layers import pointer
import utils


class Model(nn.Module):
    """ r-net for machine comprehension """
    def __init__(self, param):
        super(Model, self).__init__()

        self.w2v_size = param['embedding'].shape[1]
        self.vocab_size = param['embedding'].shape[0]
        self.embedding_type = param['embedding_type']
        self.embedding_is_training = param['embedding_is_training']
        self.mode = param['mode']
        self.hidden_size = param['hidden_size']
        self.dropout_p = param['dropout_p']
        self.encoder_dropout_p = param['encoder_dropout_p']
        self.encoder_bidirectional = param['encoder_bidirectional']
        self.encoder_layer_num = param['encoder_layer_num']
        self.is_bn = param['is_bn']

        # embedding
        if self.embedding_type == 'standard':
            self.embedding = embedding.Embedding(param['embedding'])
            is_bn = False
        else:
            self.embedding = embedding.ExtendEmbedding(param['embedding'])
            is_bn = True

        # encoder_c
        input_size = self.embedding.sd_embedding.embedding_dim + 6
        self.encoder_c = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=self.encoder_bidirectional,
            layer_num=self.encoder_layer_num,
            is_bn=is_bn
        )

        # encoder_q
        input_size = self.embedding.sd_embedding.embedding_dim + 4
        self.encoder_q = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.encoder_dropout_p,
            bidirectional=self.encoder_bidirectional,
            layer_num=self.encoder_layer_num,
            is_bn=is_bn
        )

        # match rnn
        input_size = self.hidden_size * 2 if self.encoder_bidirectional else self.hidden_size
        self.match_rnn = match_rnn.MatchRNN(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            gated_attention=True,
            is_bn=self.is_bn
        )

        # self matching attention
        input_size = self.hidden_size * 2
        self.self_match_attention = self_match_attention.SelfAttention(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            gated_attention=True,
            is_bn=self.is_bn
        )

        # addition_rnn
        input_size = self.hidden_size * 2
        self.addition_rnn = encoder.Rnn(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            bidirectional=True,
            dropout_p=self.dropout_p,
            layer_num=1,
            is_bn=self.is_bn
        )

        # init state of pointer
        self.init_state = pointer.AttentionPooling(
            input_size=self.hidden_size*2,
            output_size=self.hidden_size
        )

        # pointer
        input_size = self.hidden_size * 2
        self.pointer_net = pointer.BoundaryPointer(
            mode=self.mode,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout_p=self.dropout_p,
            bidirectional=True,
            is_bn=self.is_bn
        )

    def forward(self, batch):
        """
        :param batch: [content, question, answer_start, answer_end]
        :return: ans_range(2, batch_size, content_len)
        """
        content = batch[: 4]
        question = batch[4: 6]

        # mask
        content_mask = utils.get_mask(content[0])
        question_mask = utils.get_mask(question[0])

        # embedding
        content_vec = self.embedding(content, True)
        question_vec = self.embedding(question, False)

        # encode
        content_vec = self.encoder_c(content_vec, content_mask)
        question_vec = self.encoder_q(question_vec, question_mask)

        # match rnn
        hr = self.match_rnn(content_vec, content_mask, question_vec, question_mask)

        # self matching attention
        hr = self.self_match_attention(hr, content_mask)

        # aggregation
        hr = self.addition_rnn(hr, content_mask)

        # init state of pointer
        init_state = self.init_state(question_vec, question_mask)

        # pointer
        ans_range = self.pointer_net(hr, content_mask, init_state)

        return ans_range



