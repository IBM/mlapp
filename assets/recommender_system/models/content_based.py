import numpy as np
import pandas as pd


class ContentBasedModel:
    def __init__(self, data, items_attributes, *args, **kwargs):
        self.user_ratings_fill_na_value = kwargs.get('user_ratings_fill_na_value', 0)
        self.item_attributes_fill_na_value = kwargs.get('item_attributes_fill_na_value', 0)

        self.user_ratings = data
        self.items_attributes = items_attributes

    def fit(self, user_ratings=None, items_attributes=None):
        # update user ratings
        if user_ratings is not None:
            self.user_ratings = user_ratings

        # update item attributes
        if items_attributes is not None:
            self.items_attributes = items_attributes

    def predict(self):
        # Calculate number of attributes
        num_attributes = calc_num_attributes(self.items_attributes)

        # Calculating IDF
        idf = calc_inverse_document_frequency(self.items_attributes)

        # Transforming attributes by num_attributes
        items_attributes = self.items_attributes.apply(lambda x: np.divide(x, np.sqrt(num_attributes)))

        # Calculating User Profiles
        user_profiles = np.matmul(self.user_ratings.fillna(self.user_ratings_fill_na_value),
                                  items_attributes.fillna(self.item_attributes_fill_na_value))

        # Attributes x IDF
        attributes_x_idf = items_attributes.apply(lambda x: x * idf, axis=1)

        # Predictions
        predictions = np.matmul(user_profiles, attributes_x_idf.T)

        return pd.DataFrame(predictions)


def calc_num_attributes(data):
    return data.sum(axis=1)


def calc_term_frequency(data):
    return data.sum()


def calc_inverse_document_frequency(data):
    return np.divide(1.0, calc_term_frequency(data))
