import pandas as pd
import numpy as np
from models.recommender_system.helpers.recommender_system_evaluator \
    import transform_predictions, calculate_ranking, score_function
import scipy.optimize as optimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


class HybridModel:
    def __init__(self, num_of_users, hybrid_type='rank_agreement', weights=None, measure="auc", scaler=None,
                 weights_algorithm=None):
        self.weights = weights
        self.weights_algorithm = weights_algorithm
        self.measure = measure
        self.scaler = scaler
        self.hybrid_type = hybrid_type
        self.predictions = {
            'ranking': pd.DataFrame(),
            'score': pd.DataFrame()
        }
        self.num_of_users = num_of_users

    def fit(self, models, train=None, test=None, cv_index=None, reset_weights=False, black_list=None):
        if reset_weights:
            self.weights = None
            self.scaler = None

        accuracy_weights = []
        self.predictions = {
            'ranking': pd.DataFrame(),
            'score': pd.DataFrame()
        }

        for model_key in models:
            # get ranking prediction from each model (either from train or specific cross validation)
            for prediction_type in self.predictions:
                if cv_index is None:
                    self.predictions[prediction_type][model_key] = transform_predictions(
                        models[model_key]['predictions'], self.num_of_users,
                        transform_type=prediction_type)
                else:
                    self.predictions[prediction_type][model_key] = transform_predictions(
                        models[model_key]['cv_predictions'][cv_index], self.num_of_users,
                        transform_type=prediction_type)

            # get weight of each model by it's accuracy
            accuracy_weights.append(models[model_key]['accuracy'])

        # weight of each rank list
        if self.weights is None:
            if self.weights_algorithm == 'accuracy':
                weights_sum = np.sum([w[self.measure] for w in accuracy_weights])
                self.weights = [w[self.measure]/weights_sum for w in accuracy_weights]
            elif self.weights_algorithm == 'regression':
                # linear regression
                self.weights = self.get_linear_regression_coefficients(
                    self.predictions['score'], train)
            elif self.weights_algorithm == 'optimization':
                self.weights = self.scores_minimize_optimization(
                    self.predictions['ranking'], accuracy_weights, train, test, black_list)
            else:
                self.weights = [1 for _ in accuracy_weights]

        # scores combination
        if self.weights_algorithm == 'regression':
            self.predictions['ranking']['buckets'] = self.predictions['score'].apply(
                lambda x: self.score_combine(x, self.weights), axis=1
            )
        # agreement rank list
        else:
            self.predictions['ranking']['buckets'] = self.predictions['ranking'].apply(
                lambda x: self.rank_agreement(x, self.weights), axis=1
            )

    def predict(self):
        return self.predictions['ranking']['buckets']

    @staticmethod
    def rank_agreement(list_of_rank_lists, *args, **kwargs):
        weights = args[0]
        items = {}
        num_of_items = len(list_of_rank_lists[0])
        for idx in range(num_of_items):
            items[idx] = 0

        for r_idx, rank_list in enumerate(list_of_rank_lists):
            for idx, val in enumerate(rank_list):
                items[val] += weights[r_idx] * (num_of_items - idx)

        return items

    def score_combine(self, list_of_score_lists, *args, **kwargs):
        weights = args[0]
        items = {}
        num_of_items = len(list_of_score_lists[0])
        for idx in range(num_of_items):
            items[idx] = 0

        for r_idx, score_list in enumerate(list_of_score_lists):
            for idx, val in enumerate(score_list):
                # scale value by self.scaler
                if self.scaler is not None:
                    scaled_val = self.scaler.scale_[r_idx] * val + self.scaler.min_[r_idx]
                else:
                    scaled_val = val
                items[idx] += weights[r_idx] * scaled_val

        return items

    def scores_minimize_optimization(self, predictions, weights, train, test, black_list=None):
        def f(params):
            results = predictions.apply(
                lambda x: HybridModel.rank_agreement(x, params), axis=1
            )
            return (-1) * score_function("ranking", results, train, test, black_list, **{})[self.measure]

        initial_guess = np.array([1 for w in weights])
        result = optimize.minimize(f, initial_guess, method='SLSQP')
        return result['x']

    def get_linear_regression_coefficients(self, scores_df, train):
        model = LinearRegression(fit_intercept=False)
        predictions = pd.DataFrame()

        non_null_indices = train.apply(lambda x: [idx for idx, val in enumerate(x) if pd.notnull(val)], axis=1)

        # transforming scoring series into vectors of only non null indices
        for col in scores_df.columns:
            predictions[col] = [item for sublist in scores_df[[col]].reset_index().apply(
                lambda x: [x[col][i] for i in non_null_indices.iloc[(x['index' if 'index' in x else 'user'])]], axis=1)
                .values for item in sublist]

        if self.scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(predictions)

        model.fit(self.scaler.transform(predictions), [1 for _ in range(len(predictions))])
        return list(model.coef_)

    @staticmethod
    def cascade(list_of_scores_lists, scores=None, *args, **kwargs):
        # TODO: finish
        pass
        # remaining models for re-organize scores
        # if len(list_of_scores_lists) > 0:
        #     # init scores
        #     if scores is None:
        #         scores = copy.deepcopy(list_of_scores_lists[0])
        #
        #     std = np.std(scores)
        #     sequences = get_close_score_sequences(scores, std)
        #     final_scores = []
        #     for seq in sequences:
        #         final_scores += HybridModel.cascade(list_of_scores_lists[1:], scores[seq[0]:seq[1]])
        #     return final_scores
        # else:
        #     return scores


HYBRID_TYPES = {
    'rank_agreement': {
        'transformation': transform_predictions,
        'apply_function': HybridModel.rank_agreement,
        'kwargs': {'transform_type': 'ranking'}
    },
    'cascade': {
        # TODO:
    }
}
