import pandas as pd
import numpy as np
import implicit
import scipy.sparse as sparse

MODELS = {
    "als":  implicit.als.AlternatingLeastSquares,
    "nmslib_als": implicit.approximate_als.NMSLibAlternatingLeastSquares,
    "annoy_als": implicit.approximate_als.AnnoyAlternatingLeastSquares,
    "faiss_als": implicit.approximate_als.FaissAlternatingLeastSquares,
    "tfidf": implicit.nearest_neighbours.TFIDFRecommender,
    "cosine": implicit.nearest_neighbours.CosineRecommender,
    "bpr": implicit.bpr.BayesianPersonalizedRanking,
    "bm25": implicit.nearest_neighbours.BM25Recommender
}


# https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
class ImplicitModel:
    def __init__(self, data, *args, **kwargs):
        # preparing sparse data
        tmp_df = data.unstack().reset_index().rename(columns={'User': 'user', 'level_0': 'item', 0: 'purchases'})
        tmp_df = tmp_df.dropna()
        tmp_df['user'] = tmp_df['user'].astype("category")
        tmp_df['item'] = tmp_df['item'].astype("category")
        tmp_df['user_id'] = tmp_df['user'].cat.codes
        tmp_df['item_id'] = tmp_df['item'].cat.codes

        self.users = tmp_df[['user', 'user_id']].drop_duplicates().reset_index().sort_values('user_id').set_index('user_id')['user']
        self.items = tmp_df[['item', 'item_id']].drop_duplicates().reset_index().sort_values('item_id').set_index('item_id')['item']

        # Using the number of purchases as the ratings for the confidence + preference of the
        # ALS (Alternating Least Squares) model
        self._sparse_item_user = sparse.csr_matrix(
            (tmp_df['purchases'].astype(float), (tmp_df['item_id'], tmp_df['user_id']))
        )
        self._sparse_user_item = sparse.csr_matrix(
            (tmp_df['purchases'].astype(float), (tmp_df['user_id'], tmp_df['item_id']))
        )

        self._model = None

    def fit(self, *args, **kwargs):
        # Initialize the model
        model_name = kwargs.get('model', 'als')
        model_class = MODELS[model_name]
        alpha_val = kwargs.get('alpha_val', 15)

        if issubclass(model_class, implicit.als.AlternatingLeastSquares):
            params = {
                'factors': kwargs.get('factors', 20),
                'iterations': kwargs.get('iterations', 20),
                'regularization': kwargs.get('regularization', 0.1),
                'use_gpu': kwargs.get('use_gpu', False)
            }
        elif model_name == "bm25":
            params = {'K1': kwargs.get('K1', 100), 'B': kwargs.get('B', 0.5)}
        elif model_name == "bpr":
            params = {'factors': kwargs.get('factors', 63), 'use_gpu': kwargs.get('use_gpu', False)}
        else:
            params = {}

        self._model = model_class(**params)

        # Calculate the confidence by multiplying it by our alpha value.
        data_conf = (self._sparse_item_user * alpha_val).astype('double')

        # Fit the model
        self._model.fit(data_conf)

        return self

    def predict(self):
        return pd.DataFrame(np.dot(self._model.user_factors, self._model.item_factors.T),
                            columns=list(self.items.values), index=list(self.users.values))

    def find_similar_item(self, item, n_similar):
        # getting item id
        item_id = self.items[self.items == item].index.values[0]

        # getting n similar items
        similar = self._model.similar_items(item_id, n_similar)

        # preparing response
        items = []
        for item in similar:
            idx, score = item
            items.append(self.items.loc[idx])

        return items

    def recommend(self, user):
        # getting user id
        user_id = self.users[self.users == user].index.values[0]

        # getting recommendations scores
        recommended = self._model.recommend(user_id, self._sparse_user_item)

        # preparing response
        items = []
        scores = []
        for item in recommended:
            idx, score = item
            items.append(self.items.loc[idx])
            scores.append(score)

        result = pd.DataFrame({'item': items, 'score': scores})
        result['user'] = user
        return result[['user', 'item', 'score']]


