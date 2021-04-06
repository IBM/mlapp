from mlapp.managers import DataManager, pipeline
from mlapp.utils.general import save_temp_dataframe

import os
import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, TabularDataset, Iterator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from torchtext.vocab import Vectors

HUNDRED = 100
TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


class SentimentAnalysisDataManager(DataManager):

    def __init__(self, config, *args, **kwargs):
        DataManager.__init__(self, config, *args, **kwargs)
        language = config.get("language", 'english')
        self.punctuations = string.punctuation
        self.stop_words = stopwords.words(language)
        self.wordnet_lemmatizer = WordNetLemmatizer()

    # -------------------------------------- train methods -------------------------------------------
    @pipeline
    def load_train_data(self, *args):
        local_path = os.path.join(os.getcwd(), self.data_settings.get('local_file_path'))
        data = pd.read_csv(local_path, names=['target', 'text'])
        return data

    def clean_text(self, tokens):
        # removes any tokens that are ASCII punctuation characters ('!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~')
        tokens = list(filter(lambda token: token not in self.punctuations, tokens))

        # removes stop words using nltk stop words corpus
        tokens = list(filter(lambda t: t.lower() not in self.stop_words, tokens))

        # filter only for words that contain letters
        tokens = [token for token in tokens if re.search(r'[a-zA-Z]', token)]

        # lemmatization of words
        tokens = list(map(lambda token: self.wordnet_lemmatizer.lemmatize(token.lower()), tokens))

        return tokens

    def tokenize(self, text):
        # text lowercasing
        text = text.lower()

        # split text to tokens using nltk tokenizer
        tokens = [word for word in word_tokenize(text) if word.isalpha()]

        return tokens

    @pipeline
    def clean_train_data(self, data):
        return data

    @pipeline
    def transform_train_data(self, data):
        # gets data settings
        sample_data = self.data_settings.get("sample_data", None)
        text_column = self.data_settings.get("text_column", "text")
        target_column = self.data_settings.get("target_column", "target")
        pre_trained_embeddings = self.data_settings.get("pre_trained_embeddings")

        if sample_data is not None and isinstance(sample_data, int):
            data = data.head(sample_data)
            print("Data shape after sampling: ({0}, {1})".format(data.shape[0], data.shape[1]))

        data[target_column] = data[target_column].apply(lambda x: x-1)

        # calculates word max size


        split_train_ratio = self.data_settings.get('split_train_ratio', .8)
        print("Split data to TRAIN %s%%, TEST %s%%" % (
            str(int(split_train_ratio * HUNDRED)), str(int((1 - split_train_ratio) * HUNDRED))))
        train_data, test_data = train_test_split(data, train_size=split_train_ratio, shuffle=True)
        train_data.insert(train_data.shape[1], "order", [str(i) for i in range(len(train_data))])  # adds order column
        test_data.insert(test_data.shape[1], "order", [str(i) for i in range(len(test_data))])  # adds order column

        # save train, validation and test data in temporary output folder
        print("Saves TRAIN VALIDATION and TEST temporary text data")
        train_csv_path = save_temp_dataframe(train_data[["order", text_column]], TRAIN)
        test_csv_path = save_temp_dataframe(test_data[["order", text_column]], VALIDATION)
        print("Done saving temporary text data")

        print("Transforming text features...")
        text = Field(sequential=True, preprocessing=self.clean_text, tokenize=self.tokenize, lower=True)
        order = Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
        train_text, test_text = TabularDataset.splits(path="",
                                                      train=train_csv_path,
                                                      test=test_csv_path,
                                                      format='csv',
                                                      fields={'text': ('text', text),
                                                              'order': ('order', order)})
        print("Done creating TRAIN, VALIDATION and TEST text datasets")
        if pre_trained_embeddings is not None:
            print("Using pre trained embeddings %s" % str(pre_trained_embeddings))
            vectros = Vectors(pre_trained_embeddings)
            print("Building vocabulary...")
            text.build_vocab(train_data, test_data, vectors=vectros)
        else:
            print("Building vocabulary...")
            text.build_vocab(train_data, test_data)
        print("Done building vocabulary")
        print("Vocabulary size: %s" % str(len(text.vocab)))

        # get target data of train and test
        train_target = train_data[target_column]
        test_target = test_data[target_column]

        result = {}
        result["train_target"] = train_target
        result["test_target"] = test_target
        result["train_text"] = train_text
        result["test_text"] = test_text
        result["text"] = text
        result["order"] = order

        return result
