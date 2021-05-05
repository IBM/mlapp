from mlapp.managers import ModelManager, pipeline
from assets.sentiment_analysis.lstm_classifier import LstmClassifier
from assets.sentiment_analysis.bert_classifier import BertClassifier
from mlapp.utils.features.torch import train, evaluate, predict

import torch
import pandas as pd

# default values:
SEED_DEFAULT = 1234
BATCH_SIZE_DEFAULT = 20
EPOCHS_DEFAULT = 10
EMBEDDINGS_DEFAULT = 100
MAX_WORD_SIZE_DEFAULT = 1e3
SPLIT_TRAIN_PERCENTAGE_DEFAULT = 0.7
SPLIT_VALIDATION_PERCENTAGE_DEFAULT = 0.2
SPLIT_TEST_PERCENTAGE_DEFAULT = 0.1
STORE_CHECKPOINT_DEFAULT = False
HIDDEN_LAYERS_SIZE_DEFAULT = 1
BIDIRECTIONAL_DEFAULT = False
HIDDEN_DIM_DEFAULT = 100
NUM_LSTM_LAYERS_DEFAULT = 1
DROPOUT_DEFAULT = 0
SHUFFLE_DEFAULT = True


class SentimentAnalysisModelManager(ModelManager):

    def __init__(self, *args, **kwargs):
        ModelManager.__init__(self, *args, **kwargs)
        self.classifiers = {
            "lstm": LstmClassifier,
            "bert": BertClassifier
        }

    @pipeline
    def train_model(self, data):
        # extract data from data manager
        text = data.get("text")
        order = data.get("order")
        train_text = data.get("train_text")
        train_target = data.get("train_target")
        test_text = data.get("test_text")
        test_target = data.get("test_target")

        # creates model parameters
        output_size = self.model_settings.get('output_size')
        model_kwargs = {}
        model_kwargs['embeddings_dim'] = self.model_settings.get('embeddings_dim', EMBEDDINGS_DEFAULT)
        model_kwargs['vocab_size'] = len(text.vocab)
        model_kwargs['output_size'] = output_size
        model_kwargs['hidden_layers_size'] = self.model_settings.get('hidden_layers_size', HIDDEN_LAYERS_SIZE_DEFAULT)
        model_kwargs['bidirectional'] = self.model_settings.get('bidirectional', BIDIRECTIONAL_DEFAULT)
        model_kwargs['hidden_dim'] = self.model_settings.get('hidden_dim', HIDDEN_DIM_DEFAULT)
        model_kwargs['num_lstm_layers'] = self.model_settings.get('num_lstm_layers', NUM_LSTM_LAYERS_DEFAULT)
        model_kwargs['dropout'] = self.model_settings.get('dropout', DROPOUT_DEFAULT)
        model_kwargs['batch_size'] = self.model_settings.get('batch_size', BATCH_SIZE_DEFAULT)

        # creates model instance
        print("Creating %s model" % self.model_settings.get('classifier_type', "lstm"))
        model = self.classifiers[self.model_settings.get('classifier_type', "lstm")](**model_kwargs)

        # convert data to tensors
        train_target_tensor = torch.tensor(train_target.values)
        test_target_tensor = torch.tensor(test_target.values)

        # create data for train
        trainer_data = {}
        trainer_data['train'] = (train_text, train_target_tensor)
        trainer_data['test'] = (test_text, test_target_tensor)

        # creates model train parameters
        model_param_kwargs = {}
        model_param_kwargs['epochs'] = self.model_settings.get('epochs', EPOCHS_DEFAULT)
        model_param_kwargs['order_column'] = "order"
        model_param_kwargs['batch_size'] = self.model_settings.get('batch_size', BATCH_SIZE_DEFAULT)
        model_param_kwargs['seed'] = self.model_settings.get('seed', SEED_DEFAULT)
        model_param_kwargs['to_shuffle'] = self.model_settings.get('to_shuffle', SHUFFLE_DEFAULT)

        # train model
        print('Start training...')
        train_result = train(model, trainer_data, **model_param_kwargs)
        print('Done training')

        print('Test model on test set')
        train_result['test_results'] = evaluate(model, trainer_data['test'], **model_param_kwargs)

        # adds vocabulary to result
        train_result["vocabulary"] = text.vocab
        train_result["order"] = order

        for k, v in train_result.items():
            self.save_object(k, v)

    @pipeline
    def forecast(self, data):
        # extract data from data manager
        target_data = data.get("target_data")
        text_data = data.get("text_data")

        # getting model type
        model_type = self.model_settings.get('classifier_type', "lstm")

        print(" Loading model...")
        model = self.classifiers.get(model_type, "lstm")(**self.get_object("model_params"))
        model.load_state_dict(self.get_object("model"))

        # convert data to tensors
        target_tensor = torch.tensor(target_data.values)

        print('> Predicting...')
        model_param_kwargs = {}
        model_param_kwargs['batch_size'] = self.model_settings.get('batch_size', BATCH_SIZE_DEFAULT)
        model_param_kwargs['seed'] = self.model_settings.get('seed', SEED_DEFAULT)
        result = predict(model, (text_data, target_tensor), **model_param_kwargs)
        print('> Done predicting...')

        # saving predictions
        self.save_dataframe(pd.DataFrame(result["y_pred"], columns=['y_hat']))