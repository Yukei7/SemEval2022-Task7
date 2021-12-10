import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import numpy as np


class LSTMBase(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 n_layers=2,
                 bidirectional=True
                 ):

        """LSTM base network
        """

        super(LSTMBase, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=-1)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0
        )
        self.bidirectional = bidirectional

    def forward(self, sentence, sentence_lens):
        # sentence: [batch_size, len]
        batch_size, _, _ = sentence.size()

        # 1. embedding, if not using pretrained embedding
        # sentence = self.embedding(sentence)

        # 2. pack the padded items
        sentence = pack_padded_sequence(sentence, sentence_lens, batch_first=True, enforce_sorted=False)

        # 3. go through lstm
        lstm_out, _ = self.lstm(sentence)

        # 4. unpack the items
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, padding_value=-1)

        return lstm_out


class LSTMClassifier(LSTMBase):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 target_size=2,
                 n_layers=2,
                 bidirectional=True):
        super().__init__(input_dim, embedding_dim, hidden_dim, n_layers, bidirectional)
        self.target_size = target_size
        # binary
        self.hidden2tag = nn.Linear(self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
                                    self.target_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence, sentence_lens):
        lstm_out = super(LSTMClassifier, self).forward(sentence, sentence_lens)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.softmax(tag_space)
        return tag_scores


class LSTMRegressor(LSTMBase):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 n_layers=2,
                 bidirectional=True):
        super(LSTMRegressor, self).__init__(input_dim, embedding_dim, hidden_dim, n_layers, bidirectional)

        # regression
        self.linear = nn.Linear(in_features=self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
                                out_features=1)

    def forward(self, sentence, sentence_lens):
        lstm_out = super(LSTMRegressor, self).forward(sentence, sentence_lens)
        out = self.linear(lstm_out)
        return out


class LSTMMulti(LSTMBase):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 target_size=2,
                 n_layers=2,
                 bidirectional=True):
        super(LSTMMulti, self).__init__(input_dim, embedding_dim, hidden_dim, n_layers, bidirectional)
        self.target_size = target_size
        # binary
        self.hidden2tag = nn.Linear(in_features=self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
                                    out_features=self.target_size)

        # regression
        self.regression = nn.Linear(in_features=self.hidden_dim * 2 if self.bidirectional else self.hidden_dim,
                                    out_features=1)

    def forward(self, sentence, sentence_lens):
        lstm_out = super(LSTMMulti, self).forward(sentence, sentence_lens)
        tag = self.hidden2tag(lstm_out)
        val = self.regression(lstm_out)
        # Tensor(batch, len, tag) + Tensor(batch, len, val)
        # Tensor(batch, len, 3)
        return torch.cat((tag, val), -1)


if __name__ == "__main__":
    input = np.random.random(size=(4, 4, 128))
    lstm = LSTMMulti(input_dim=128, embedding_dim=128, hidden_dim=64)
    lstm.train()
    lstm(torch.tensor(input).float(), [4, 3, 2, 1])
