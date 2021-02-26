import inspect
import pandas as pd
import numpy as np
from sklearn.metrics import *
from sklearn.feature_selection import *
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *
from mlapp.utils.metrics.pandas import classification, regression, time_series
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import make_scorer
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, BayesianRidge, ElasticNet, Lars, LassoLars, \
    LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from mlapp.utils.exceptions.framework_exceptions import AutoMLException
from mlapp.utils.auto_ml.base import _AutoMLBase
from mlapp.utils.visualizations.pandas import visualization_methods

try:
    import xgboost as xgb
except ModuleNotFoundError as e:
    xgb = None
try:
    import lightgbm as lgb
except ModuleNotFoundError as e:
    lgb = None


class _AutoMLPandas(_AutoMLBase):
    def _validate_input(self, estimator_family, train, test, *args, **kwargs):
        if estimator_family not in ['linear', 'non_linear', 'binary', 'multi_class', 'clustering']:
            raise AutoMLException("ERROR: model family not supported.")

        for data in [train, kwargs['y_train'], test, kwargs['y_test']]:
            if not (isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) or isinstance(data, np.ndarray)):
                raise AutoMLException("ERROR: data type should be Pandas Series/Dataframe or Numpy ndarray.")

        if train.shape[0] <= 1:
            raise AutoMLException("ERROR: Use more than 1 column for your train and test set!")

        if kwargs['search_type'] not in ['grid', 'random']:
            raise AutoMLException("ERROR: 'search_type' must be 'grid' or 'random'!")

        # TODO: add more validations tests

    def _eval_config_input(self, config_input):
        for key in config_input:
            if isinstance(config_input[key], dict):
                # go in one level
                self._eval_config_input(config_input[key])
            else:
                try:
                    # try to eval str
                    if isinstance(config_input[key], str):
                        config_input[key] = eval(config_input[key])
                except:
                    continue

        return config_input

    def _default_model_classes(self, estimator_family):
        return {
            'linear': {
                'Lasso': Lasso,
                'LinearRegression': LinearRegression,
                'Ridge': Ridge,
                'ElasticNet': ElasticNet,
                'Lars': Lars,
                'LassoLars': LassoLars,
                'BayesianRidge': BayesianRidge
            },
            'non_linear': {
                'RandomForest': RandomForestRegressor,
                'ExtraTree': ExtraTreesRegressor,
                'XGBoost': xgb.XGBRegressor if xgb else None,
                'LightGBM': lgb.LGBMRegressor if lgb else None,
                'GradientBoosting': GradientBoostingRegressor if not xgb and not lgb else None
            },
            'binary': {
                'Logistic': LogisticRegression,
                'RBF SVM': SVC,  # binary
                'MultiLogistic': LogisticRegression,
                'ExtraTreeClassifier': ExtraTreesClassifier,
                'RandomForest': RandomForestClassifier,
                'GaussianNB': GaussianNB,
                'GaussianProcess_Classifier': GaussianProcessClassifier,
                'KnearestNeighbors': KNeighborsClassifier,
                'XGBoostClassifier': xgb.XGBClassifier if xgb else None,
                'LightGBMClassifier': lgb.LGBMClassifier if lgb else None,
                'GradientBoostingClassifier': GradientBoostingClassifier if not xgb and not lgb else None,
                'AdaBoostClassifier': AdaBoostClassifier
            },
            'multi_class': {
                'Logistic': LogisticRegression,
                'RBF SVM': SVC,  # binary
                'MultiLogistic': LogisticRegression,
                'ExtraTreeClassifier': ExtraTreesClassifier,
                'RandomForest': RandomForestClassifier,
                'GaussianNB': GaussianNB,
                'GaussianProcess_Classifier': GaussianProcessClassifier,
                'KnearestNeighbors': KNeighborsClassifier,
                'XGBoostClassifier': xgb.XGBClassifier if xgb else None,
                'LightGBMClassifier': lgb.LGBMClassifier if lgb else None,
                'GradientBoostingClassifier': GradientBoostingClassifier if not xgb and not lgb else None,
                'AdaBoostClassifier': AdaBoostClassifier
            },
            'clustering': {
                'AgglomerativeClustering': AgglomerativeClustering,
                'KMeans': KMeans,
                'MiniBatchKmeans': MiniBatchKMeans,
                'DB-SCAN': DBSCAN,
                'MeanShift': MeanShift,
                'SpectralClustering': SpectralClustering
            }
        }.get(estimator_family, {})

    def _default_fixed_params(self, estimator_family):
        return {
            # TODO: add default fixed params
        }.get(estimator_family, {})

    def _default_hyper_params(self, estimator_family):
        return {
            # TODO: add default hyper params
        }.get(estimator_family, {})

    def _visualization_methods(self, method):
        return visualization_methods(method)

    def _get_model_intercept_and_coefficients(self, model, features):
        try:
            if len(model.coef_) == 1:
                coefficients = {features[index]: model.coef_[0][index] for index, feature in
                                enumerate(model.coef_[0])}
            else:
                coefficients = {features[index]: model.coef_[index] for index, feature in enumerate(model.coef_)}

            return model.intercept_, coefficients
        except:
            try:
                return 0, {feature: model.feature_importances_[index] for index, feature in enumerate(features)}
            except:
                try:
                    return 0, model.booster().get_fscore()
                except:
                    try:
                        return 0, model._Booster.get_fscore()
                    except:
                        print("Warning: not able to get model coefficients.")
                        return 0, []

    def _calc_cv_df(self, searches, key, cv_weights):
        # get splits
        splits = list(filter(lambda x: True if all(word in x for word in ['split', 'test_score']) else False,
                             searches[key].cv_results_.keys()))
        splits_df = pd.DataFrame(np.transpose([searches[key].cv_results_[x] for x in splits]), columns=splits)

        # create column for each hyper param
        hyper_params_attr = list(filter(
            lambda x: True if all(word in x for word in ['param_']) else False, searches[key].cv_results_.keys()))
        cv_results = pd.DataFrame()
        for column_name in hyper_params_attr:
            cv_results[column_name] = searches[key].cv_results_[column_name]

        # calculate hyper param score
        cv_results['max_score'] = splits_df.apply(lambda x: max(x), axis=1)
        cv_results['min_score'] = splits_df.apply(lambda x: min(x), axis=1)
        cv_results['mean_score'] = splits_df.apply(lambda x: np.mean(x), axis=1)
        cv_results['std_score'] = splits_df.apply(lambda x: np.std(x), axis=1)
        cv_results['estimator'] = key

        # add weighted mean score
        if not cv_weights:
            cv_weights = 1
        cv_results['weighted_mean_score'] = splits_df.apply(lambda x: np.mean(np.multiply(x, cv_weights)), axis=1)

        return cv_results

    def _get_cv_results(self, estimator_family, searches, features, scoring, greater_is_better, *args, **kwargs):
        """
        Function to get the best model in the GridSearchCV analysis.
        The analysis is based on the model that performs the best the score on the validation sets from the cross validation
        Indeed, during the cross validation, each set will be consecutively the validations set and obtains a score (from
        the scorer). Then, we average all those scores and the model that brings the higher (or lower) score is considered
        the best model.
        :param keys: all GS keys
        :param searches: actual results of the GS
        :param features: coefficients of the model
        :param sort_by: CV score to sort by (max_score, min_score, mean_score, std_score, weighted_mean_score)
        :param low_score_is_better: boolean - whether low score is better
        :param cv_weights: the weights of the CV splits
        :return: dataframe of features, name of the best model, instance of the best model, intercept, coefficients
        """
        sort_by = kwargs.get('sort_by', 'weighted_mean_score')

        # get cv results df
        df = pd.concat([self._calc_cv_df(searches, key, kwargs['cv_weights']) for key in searches.keys()], sort=False)\
            .sort_values(by=sort_by, ascending=False)

        # reorder columns
        columns = ['estimator', 'min_score', 'mean_score', 'weighted_mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        df = df[columns]

        # fetch first row results => best result
        best_model_key = df.reset_index()['estimator'][0]
        best_model = searches[df.reset_index()['estimator'][0]].best_estimator_
        best_cv_score = df.iloc[0]['mean_score']
        intercept, coefficients = self._get_model_intercept_and_coefficients(best_model, features)

        return df, best_model_key, best_model, best_cv_score, intercept, coefficients

    def _predict_train_test(self, model, train, test, *args, **kwargs):
        train_y_hat = model.predict(train).reshape(1, len(train))[0]
        test_y_hat = model.predict(test).reshape(1, len(test))[0]
        return train_y_hat, test_y_hat

    def _full_data_model_fit(self, best_model, train, test, *args, **kwargs):
        best_model.fit(pd.concat([train, test]), np.concatenate((kwargs['y_train'], kwargs['y_test'])))
        return best_model

    def _feature_selection_globals(self, method):
        if method in globals():
            return globals()[method]

    def _run_feature_selection(self, feature_selection, train, test, *args, **kwargs):
        feature_selection_clf = self._load_feature_selection_method(feature_selection.get('method', ''))
        evaluated_params = self._eval_config_input(feature_selection.get('params', {}))

        if not feature_selection_clf:
            return list(train.columns), train, test

        feature_selection_method = feature_selection_clf(**evaluated_params)
        feature_selection_method.fit(train, kwargs.get('y_train'))

        selected_features = list(train.columns[feature_selection_method.get_support()])
        return selected_features, train[selected_features], test[selected_features]

    def _get_scorer(self, estimator_family, metric=None, *args, **kwargs):
        if metric:
            if callable(metric):
                if isinstance(metric, _BaseScorer):
                    return metric
                else:
                    return make_scorer(metric, greater_is_better=kwargs.get('greater_is_better', True))
            else:
                return make_scorer(globals()[metric], greater_is_better=kwargs.get('greater_is_better', True))
        else:
            return {
                'linear': make_scorer(mean_squared_error, greater_is_better=False),
                'non_linear': make_scorer(mean_squared_error, greater_is_better=False),
                'multi_class': make_scorer(f1_score, average='weighted'),
                'binary': make_scorer(f1_score, average='binary')
            }.get(estimator_family)

    def _scores_function(self, scores_summary_func, estimator_family):
        # take scores summary by 'scores_summary' then by 'best_estimator'
        return {
            'classification': classification,
            'regression': regression,
            'time_series': time_series
        }.get(scores_summary_func, {
            'linear': regression,
            'non_linear': regression,
            'multi_class': classification,
            'binary': classification
        }.get(estimator_family))

    def _inner_search(self, estimator_family, train, test, model_key, model_class, fixed_params, hyper_params, scoring,
                      greater_is_better, cv, *args, **kwargs):
        model_object = model_class(**fixed_params.get(model_key, {}))
        tuned_params = hyper_params.get(model_key, {})
        auto_search_params = {
            'cv': cv,
            'verbose': 1,
            'refit': True,
            'scoring': self._get_scorer(estimator_family, scoring, greater_is_better=greater_is_better, *args, **kwargs),
            'n_jobs': kwargs.get('n_jobs', 1),
            'return_train_score': True
        }
        if kwargs['search_type'] == 'grid' or len(tuned_params.keys()) == 0:
            auto_search = GridSearchCV(model_object, tuned_params, **auto_search_params)
        else:
            auto_search = RandomizedSearchCV(model_object, tuned_params, **auto_search_params)

        auto_search.fit(train, kwargs['y_train'])

        return auto_search
