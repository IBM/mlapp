import pandas as pd
import numpy as np
from common.data_science.dataframe_utilites import calc_cosine_similarity


class ItemItemCollaborativeFilteringModel:
    def __init__(self, ratings, *args, **kwargs):
        # init values
        self.is_unary_data = kwargs.get('is_unary_data', False)
        self.components = {}
        self.components_keys = {
            "ratings", "normalized_ratings", "item_mean_ratings", "user_ratings_mean", "cosine_similarity"
        }

        self.components = self._calc_item_item_cf_components(ratings)

    def fit(self):
        pass

    def refit(self, ratings):
        self.components = self._calc_item_item_cf_components(ratings, prev_calc=True)

    def predict(self, user=None):
        # calculating all users
        if user is None:
            ratings_df = self.components['ratings']
            columns = list(self.components['ratings'].columns)
            indexes = list(self.components['ratings'].index)
            axis = 1
        # calculating only for specific user
        else:
            ratings_df = self.components['ratings'].loc[user]
            columns = [user]
            indexes = list(self.components['ratings'].columns)
            axis = 0
        ratings_x_similarity = pd.DataFrame(
            np.matmul(ratings_df, self.components['cosine_similarity']),
            columns=columns)
        ratings_x_similarity.index = indexes

        # check if data is unary
        if self.is_unary_data:
            scores = ratings_x_similarity
        else:
            item_cosine_similarity_sum = self.components['cosine_similarity'].sum()
            scores = ratings_x_similarity.apply(lambda x: np.divide(x, item_cosine_similarity_sum), axis=1)

        scores = scores.apply(lambda x: self.components['item_mean_ratings']['value'] + x, axis=axis)

        return scores

    def find_similar_item(self, item, n_similar):
        return self.components['cosine_similarity'][item].nlargest(n_similar)

    def recommend(self, user):
        return self.predict(user)

    def _calc_item_item_cf_components(self, ratings, prev_calc=False):
        # Removing previous calculations of updated user-item ratings
        if prev_calc:
            items_to_drop = [item for item in list(ratings.columns) if item in list(self.components['ratings'].columns)]
            if len(items_to_drop) > 0:
                for (component, axis) in [(component, axis) for component in self.components_keys for axis in [0, 1]]:
                    try:
                        self.components[component].drop(labels=items_to_drop, axis=axis, inplace=True)
                    except KeyError:
                        pass
                # TODO: decide if to do something with 'user_ratings_mean'

        # Normalizing users ratings
        user_ratings_mean = pd.concat([
            ratings.mean(axis=1),
            np.sum(ratings.agg(pd.notnull, axis=1), axis=1)], axis=1)\
            .rename(columns={0: 'value', 1: 'count'})
        item_ratings_mean = pd.DataFrame(ratings.mean(), columns=['value'])
        if self.components.get('user_ratings_mean') is not None:
            # (old sum + new sum) / (old count + new count)
            user_ratings_mean['value'] = \
                (user_ratings_mean['value'].fillna(0) * user_ratings_mean['count'] +
                 self.components['user_ratings_mean']['value'] * self.components['user_ratings_mean']['count']) / \
                (user_ratings_mean['count'] + self.components['user_ratings_mean']['count'])
            user_ratings_mean['count'] = \
                (user_ratings_mean['count'] + self.components['user_ratings_mean']['count'])

        # Filling 0s
        ratings = ratings.fillna(0)

        # TODO: decide if to update previous ratings as well or redundant or make it configurable?
        ratings = ratings.apply(lambda x: x - user_ratings_mean['value'])

        # L2 - Normalizing ratings
        normalized_ratings_vector = pd.DataFrame(np.sqrt(np.sum(np.power(ratings, 2))), columns=['value'])

        # Previous calculation available
        if prev_calc:
            # Arranging index same as previous calculation ratings
            ratings = ratings.reindex(self.components['ratings'].index)

            # Updating vector somehow
            normalized_ratings_vector = pd.concat([self.components['normalized_ratings'], normalized_ratings_vector])

            # Updating item ratings
            item_ratings_mean = pd.concat([self.components['item_mean_ratings'], item_ratings_mean])

            self.components['ratings'] = pd.concat([self.components['ratings'], ratings], axis=1)

        # Calculate cosine similarity
        cosine_similarity_df = calc_cosine_similarity(
            ratings, normalized_ratings_vector['value'], self.components.get('ratings'))

        # Transforming values to non-negative
        cosine_similarity_df = cosine_similarity_df.applymap(lambda x: x if x > 0 else 0)

        # Merge results from this calculation and previous
        if prev_calc:
            # Merged ratings
            ratings = self.components['ratings']

            # Merging cosine similarity
            """
            A = prev cosine similarity, B = new cosine similarity
            C = Concat(A, B[0:len(A)].T, axis=0)  =>   Concat(C, B, axis=1)
                A        B      B[0:len(A)].T               C         B     
            [ 1 2 3 ] [7  8 ]     [7 9  11]            [ 1  2  3 ] [7  8 ]
            [ 2 1 5 ] [9  10]     [8 10 12]            [ 2  1  5 ] [9  10]
            [ 3 5 1 ] [11 12]                          [ 3  5  1 ] [11 12] 
                      [1  13]                          [ 7  9  11] [ 1 13]  
                      [13 1 ]                          [ 8  10 12] [14  1] 
            """
            self.components['cosine_similarity'] = pd.concat([
                self.components['cosine_similarity'],
                cosine_similarity_df[0:self.components['cosine_similarity'].shape[0]].T
            ])
            cosine_similarity_df = pd.concat([self.components['cosine_similarity'], cosine_similarity_df], axis=1)

        return {
            "ratings": ratings,
            "normalized_ratings": normalized_ratings_vector,
            "item_mean_ratings": item_ratings_mean,
            "user_ratings_mean": user_ratings_mean,
            "cosine_similarity": cosine_similarity_df
        }


