import torch
from torch import nn
import torch.nn.functional as F
from mlapp.utils.exceptions.user_exceptions import MissingArgumentsError
from transformers import BertModel, BertTokenizer

ONE = 1
HUNDRED = 100


class BertClassifier(nn.Module):

    def __init__(self, **kwargs):
        super(BertClassifier, self).__init__()

        # set network parameters
        self.vocab_size = kwargs.get("vocab_size", ONE)
        self.output_size = kwargs.get("output_size", ONE)
        self.num_lstm_layers = kwargs.get("num_lstm_layers", ONE)
        self.numeric_features_size = kwargs.get("numeric_features_size", ONE)
        self.hidden_layers_size = kwargs.get('hidden_layers_size', ONE)
        self.hidden_dim = kwargs.get('hidden_dim', HUNDRED)
        self.dropout_ratio = kwargs.get('dropout', 1)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.pre_trained_bert = kwargs.get('pre_trained_bert', 'bert-base-uncased')
        self.seed = kwargs.get('seed')
        self.num_directions = 2 if self.bidirectional else 1

        # initialize network layers
        self.bert = BertModel.from_pretrained(self.pre_trained_bert)
        self.embeddings_dim = self.bert.config.to_dict()['hidden_size']
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
        params['numeric_features_size'] = self.numeric_features_size
        params['embeddings_dim'] = self.embeddings_dim
        params['bidirectional'] = self.bidirectional
        params['num_lstm_layers'] = self.num_lstm_layers
        params['seed'] = self.seed
        return params

    def forward(self, input_data, **kwargs):
        """

        :param text: (batch_size, num_sequences)
        :return:
        """
        if not(self.vocab_size or self.output_size or self.numeric_features_size):
            raise MissingArgumentsError("vocab_size, output_size and numeric_features_size are required")


        # words embeddings
        with torch.no_grad():
            words_embeds = self.bert(input_data.text.T[:, :BertTokenizer.max_model_input_sizes[self.pre_trained_bert]])[0]  #words_embeds = [batch size, sent len, emb dim]
            words_embeds = words_embeds.permute(1, 0, 2)

        _, (hn, cn) = self.lstm(words_embeds)
        if self.bidirectional:
            hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        else:
            hidden = hn[-1,:,:]

        # fully-connected layers 1
        dense1 = self.fc1(hidden)  # (batch_size, output_size)

        # perform dropout if self.dropout_ratio is grater than zero
        drop = self.dropout(dense1)

        # fully-connected layers 2
        output = self.fc2(drop)

        # performing softmax on output
        output = F.softmax(output, dim=1)

        return output
