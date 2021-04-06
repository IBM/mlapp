import torch
from torch import nn
import torch.nn.functional as F
from mlapp.utils.exceptions.user_exceptions import MissingArgumentsError


class LstmClassifier(nn.Module):

    def __init__(self, **kwargs):
        super(LstmClassifier, self).__init__()

        # set network parameters
        self.vocab_size = kwargs.get("vocab_size", 1)
        self.output_size = kwargs.get("output_size", 1)
        self.batch_size = kwargs.get("batch_size", 1)
        self.num_lstm_layers = kwargs.get("num_lstm_layers", 1)
        self.embeddings_dim = kwargs.get('embeddings_dim', 100)
        self.hidden_layers_size = kwargs.get('hidden_layers_size', 1)
        self.hidden_dim = kwargs.get('hidden_dim', 100)
        self.pre_trained_embeddings = kwargs.get('pre_trained_embeddings')
        self.dropout_ratio = kwargs.get('dropout', 1)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.seed = kwargs.get('seed')
        self.num_directions = 2 if self.bidirectional else 1

        # initialize network layers
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embeddings_dim)
        if self.pre_trained_embeddings is not None:  # checks for embeddings pre trained weights initialization
            self.word_embeddings.from_pretrained(self.pre_trained_embeddings)
        self.lstm = nn.LSTM(self.embeddings_dim, self.hidden_layers_size, num_layers=self.num_lstm_layers, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.fc1 = nn.Linear(self.hidden_layers_size * self.num_directions, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_size)

        if self.seed:
            torch.manual_seed(self.seed)

    def get_params(self):
        params = {}
        params['vocab_size'] = self.vocab_size
        params['output_size'] = self.output_size
        params['hidden_layers_size'] = self.hidden_layers_size
        params['embeddings_dim'] = self.embeddings_dim
        params['pre_trained_embeddings'] = self.pre_trained_embeddings
        params['bidirectional'] = self.bidirectional
        params['num_lstm_layers'] = self.num_lstm_layers
        params['seed'] = self.seed
        return params

    def forward(self, input_data, **kwargs):


        if not(self.vocab_size or self.output_size or self.numeric_features_size):
            raise MissingArgumentsError("vocab_size and output_size are required")

        # words embeddings
        words_embeds = self.word_embeddings(input_data.text.T)  # transposing text because we gets (text_length, batch_size) and we want (batch_size, text_length)
        words_embeds = words_embeds.permute(1, 0, 2)

        # performing lstm layer, using h0 and c0 on text
        h0 = torch.zeros(self.num_directions * self.num_lstm_layers, self.batch_size, self.hidden_layers_size)
        h0 = h0.cuda() if torch.cuda.is_available() else h0
        c0 = torch.zeros(self.num_directions * self.num_lstm_layers, self.batch_size, self.hidden_layers_size)
        c0 = c0.cuda() if torch.cuda.is_available() else c0
        output, (hn, cn) = self.lstm(words_embeds, (
        h0, c0))  # output, hn - last lstm hidden state (1, batch_size, hidden_size) , cn - last lstm cell state

        # concatenate lstm text representation with features
        if self.bidirectional:
            output = torch.cat((hn[-1, :, :], hn[-2, :, :]), dim=1)
            dense1 = self.fc1(output)
        else:
            # fully-connected layers 1
            dense1 = self.fc1(hn[-1])  # (batch_size, output_size)

        # perform dropout if self.dropout_ratio is grater than zero
        drop = self.dropout(dense1)

        # fully-connected layers 2
        output = self.fc2(drop)

        # performing softmax on output
        output = torch.sigmoid(output)

        return output