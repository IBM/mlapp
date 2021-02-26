import copy
import pandas as pd
from abc import ABC, abstractmethod
from mlapp.utils.auto_ml.pandas import _AutoMLPandas
try:
    from mlapp.utils.auto_ml.spark import _AutoMLSpark
except:
    _AutoMLSpark = None

from mlapp.utils.exceptions.framework_exceptions import AutoMLException


class AutoMLSettings:
    def __init__(self, metric=None, greater_is_better=True, search_type='grid', scores_summary=None):
        """
        AutoML Settings Object

        :param metric: scorer function for cross validation. By default auto-selects based on estimator type:
        'linear'/'non-linear': mean squared error
        'binary'/'multi_class': f1 score

        Accepts any scorer depending on framework. E.g `r2_score` from sklearn.metrics when using AutoMLPandas.

        :param greater_is_better: boolean, i.e. for MAPE, low is better.
        :param search_type: 'grid'/'random' - type of search to be run to find the best combination.
        :param scores_summary: 'regression'/'classification'/'time_series' or None auto-selects by estimator.
        """
        self.metric = metric
        self.scores_summary = scores_summary
        self.greater_is_better = greater_is_better
        self.search_type = search_type

    def is_same_settings(self, s: 'AutoMLSettings'):
        return (self.metric == s.metric or self.metric is s.metric) and \
               self.greater_is_better == s.greater_is_better and \
               self.scores_summary == s.scores_summary


class AutoMLResults:
    """
    AutoMLResults Object contains results of AutoML run/s.

    :param settings: AutoMLSettings Object
    """
    def __init__(self, settings=AutoMLSettings()):
        self.settings = settings
        self.cv_metrics = pd.DataFrame()
        self.best_cv_score = None
        self.best_model_metrics = {}
        self.best_model = None
        self.selected_features = []
        self.intercept = None
        self.coefficients = []
        self.figures = {}
        self.best_estimator = None
        self.best_model_key = None
        self.best_feature_selection = None
        self.best_train_predictions = None
        self.best_test_predictions = None
        self.scores_function = None

    def add_cv_run(self, scores_function, best_estimator, feature_selection, train_predicted, test_predicted,
                   selected_features, best_model, intercept, coefficients, cv_score, all_cv_scores, figures,
                   *args, **kwargs):

        # assign feature selection column
        if 'feature_selection' not in all_cv_scores.columns:
            all_cv_scores.insert(loc=1, column='feature_selection', value=feature_selection)

        self.cv_metrics = pd.concat([self.cv_metrics, all_cv_scores])   # append cv metrics

        # new best score
        if not self.best_cv_score or cv_score > self.best_cv_score:

            # overwrite best model results
            if kwargs.get('model_metrics'):
                self.best_model_metrics = kwargs['model_metrics']
            else:
                self.best_model_metrics = self._scores_summary(
                    scores_function, train_predicted, test_predicted, *args, **kwargs)

            self.best_model = best_model
            self.best_train_predictions = train_predicted
            self.best_test_predictions = test_predicted
            self.best_cv_score = cv_score
            self.intercept = intercept
            self.selected_features = selected_features
            self.coefficients = coefficients
            self.figures = figures
            self.best_model_key = best_model.__class__.__name__
            self.best_estimator = best_estimator
            self.best_feature_selection = feature_selection
            self.scores_function = scores_function

    def merge(self, results: 'AutoMLResults'):
        if not AutoMLResults.is_same_settings(self, results):
            raise AutoMLException("Error: Tried to merge AutoML runs results with different settings!")

        self.add_cv_run(None, results.best_estimator, results.best_feature_selection,
                        results.best_train_predictions, results.best_test_predictions,
                        results.selected_features, results.best_model, results.intercept, results.coefficients,
                        results.best_cv_score, results.cv_metrics, results.figures,
                        model_metrics=results.best_model_metrics)

    def get_best_model(self):
        """
        Gets best model from AutoML run.
        """
        return self.best_model

    def get_metrics(self):
        """
        Gets metrics result associated with the best model.
        """
        return copy.copy(self.best_model_metrics)

    def get_figures(self):
        """
        Gets figures produced by AutoML run associated with the best model.
        """
        return self.figures

    def print_report(self, full=True, ascending=False):
        """
        Prints score report of all runs done by the AutoML.

        :param full: prints scores from all model runs, or use just best representative from each model (True/False)
        :param ascending: show scores in ascending order (True/False)
        """
        with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.expand_frame_repr', False,
                'max_colwidth', -1):
            df = self.cv_metrics.sort_values(by='mean_score', ascending=ascending)
            if not full:
                df = df.drop_duplicates(['estimator'])
            print("------------------------------------ AutoML Report -------------------------------------------")
            print(df)
            print("----------------------------------------------------------------------------------------------")

    def get_metadata(self):
        """
        Gets all metadata associated with the best model:

        > scores: scores associated with estimator used
        > estimator_family: estimator family (linear/non-linear/classificaton/etc.)
        > model: best model class name
        > selected_features_names: selected features
        > intercept: intercept of model if exists
        > coefficients: coefficients / feature importance if exist
        > feature_selection: feature selection method
        > cv_score: cross validation score
        """
        return {
            "scores": self.get_metrics(),
            "estimator_family": self.best_estimator,
            "model_class_name": self.best_model_key,
            "selected_features_names": self.selected_features,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "feature_selection": self.best_feature_selection,
            "cv_score": self.best_cv_score
        }

    def get_train_predictions(self):
        """
        Gets train predictions from the best model.
        """
        return self.best_train_predictions

    def get_test_predictions(self):
        """
        Gets test predictions from the best model.
        """
        return self.best_test_predictions

    def get_selected_features(self):
        """
        Gets selected features associated with the best model.
        """
        return self.selected_features

    def predict_proba_by_threshold(self, x_train, x_test, y_train, y_test, threshold):
        """
        Prints score report of all runs done by the AutoML.

        :param x_train: train data (pd.DataFrame)
        :param x_test: test data (pd.DataFrame)
        :param y_train: train data (pd.DataFrame/pd.Series)
        :param y_test: test data (pd.DataFrame/pd.Series)
        :param threshold: Threshold percentage/s to use for predict by probability ([float]/float)

        :return: dictionary containing relevant scores.
        """
        results = {}

        if hasattr(self.best_model, 'predict_proba'):
            train_predict_proba = self.best_model.predict_proba(x_train[self.selected_features])[:, 1]
            test_predict_proba = self.best_model.predict_proba(x_test[self.selected_features])[:, 1]

            if not isinstance(threshold, list):
                threshold = [threshold]

            for proba_threshold in threshold:
                train_predicted = train_predict_proba > proba_threshold
                test_predicted = test_predict_proba > proba_threshold
                results[proba_threshold] = self._scores_summary(
                    self.scores_function, self.best_train_predictions, self.best_test_predictions, y_train, y_test,
                    train_y_proba=train_predicted, test_y_proba=test_predicted)

            return results

        return "best model doesn't support 'predict_proba'"

    @staticmethod
    def is_same_settings(r1: 'AutoMLResults', r2: 'AutoMLResults'):
        return r1.settings.is_same_settings(r2.settings)

    @staticmethod
    def _scores_summary(scores_function, train_predicted, test_predicted, y_train=None, y_test=None, *args, **kwargs):
        if scores_function:
            return scores_function(train_predicted, test_predicted, y_train, y_test, *args, **kwargs)
        else:
            return {}


class _AutoMLWrapper(ABC):
    def __init__(self, estimator, settings=AutoMLSettings(), models=None, fixed_params=None, hyper_params=None,
                 model_classes=None, feature_selection=None, visualizations=None):
        if not isinstance(settings, AutoMLSettings):
            raise AutoMLException(f"Error: 'settings' must be of type: 'AutoMLSettings'.")

        self.estimator = estimator
        self.settings = settings
        self.models = models
        self.model_classes = model_classes
        self.fixed_params = fixed_params
        self.hyper_params = hyper_params
        self.feature_selection = feature_selection
        self.visualizations = visualizations
        self.auto_ml_class = None

    @abstractmethod
    def run(self, *args, **kwargs) -> 'AutoMLResults':
        pass

    def _run(self, x_train, x_test, *args, **kwargs):
        results = AutoMLResults(self.settings)
        return self.auto_ml_class().run(
            results, self.settings.scores_summary, self.feature_selection, self.estimator, x_train, x_test, self.models,
            self.fixed_params, self.hyper_params, self.model_classes, self.settings.metric,
            self.settings.greater_is_better, search_type=self.settings.search_type, visualizations=self.visualizations,
            *args, **kwargs)


class AutoMLPandas(_AutoMLWrapper):
    def __init__(self, estimator, settings=AutoMLSettings(), models=None, fixed_params=None, hyper_params=None,
                 model_classes=None, feature_selection=None, visualizations=None):
        """
        This algorithm enables data scientists to find the best models among different options with different
        parametrization. It will pick the best model among the models, including the hyper params combination
        that performed the configured scoring the best.

        :param settings: AutoMLSettings
        :param estimator: model family to use: 'linear', 'non_linear', 'binary', 'multi_class'.
        :param models: assets available in the functionality depending on estimator_family:

            linear:
                'Lasso' 'LinearRegression', 'Ridge', 'ElasticNet', 'Lars', 'LassoLars' and 'BayesianRidge'.

            non_linear:
                'RandomForest', 'ExtraTree', 'XGBoost', 'LightGBM' and 'GradientBoosting'.

            binary:
                'Logistic', 'RBF , 'MultiLogistic', 'ExtraTreeClassifier', 'RandomForest', 'GaussianNB',
                'GaussianProcess_Classifier', 'KnearestNeighbors', 'GradientBoosting', 'XGBoostClassifier',
                'LightGBMClassifier', 'GradientBoostingClassifier' and 'AdaBoostClassifier'.

            multi_class:
                'Logistic', 'RBF , 'MultiLogistic', 'ExtraTreeClassifier', 'RandomForest', 'GaussianNB',
                'GaussianProcess_Classifier', 'KnearestNeighbors', 'GradientBoosting', 'XGBoostClassifier',
                'LightGBMClassifier', 'GradientBoostingClassifier' and 'AdaBoostClassifier'.

            clustering:
                'AgglomerativeClustering', 'KMeans', 'MiniBatchKmeans', 'DB-SCAN', 'MeanShift' and 'SpectralClustering'.

        :param fixed_params: dictionary, initialize each model with these fixed params. Each key should correspond to a key
        from the assets and values should be a dictionary: {'param_name': 'param_value'}. By default, the assets will be
        run with default configuration. Please, refer to online documentation (sklearn ..) to find out what are the possible
        parameters and values.

        :param hyper_params: dictionary of hyper_params for the GridSearch explorations. Each key should correspond to a key
        from the assets and values should be a dictionary: {'param_name': [list_of_values_to_explore]}. By default, the
        assets will be run without any hyper_params. Please, refer to online documentation (sklearn ..) to find out
        what are the possible parameters and values.

        :param model_classes: dictionary of model classes to pass to be run by the AutoML.

        :param visualizations: dictionary where key is the name and value is the related function.
        Please refer to utils.visualizations to check which function are available.
        """
        if visualizations is None:
            visualizations = ["score_evolution_hyper_params"]

        super(AutoMLPandas, self).__init__(estimator, settings, models, fixed_params, hyper_params, model_classes,
                                           feature_selection, visualizations)

        self.auto_ml_class = _AutoMLPandas

    def run(self, x_train, y_train, x_test, y_test, cv=5, cv_weights=None, n_jobs=1, best_model_full_fit=False) \
            -> 'AutoMLResults':
        """
        Runs the AutoML

        :param x_train: pandas DataFrame of train set of features
        :param y_train: pandas Series or DataFrame train target.
        :param x_test: pandas DataFrame of test set of features
        :param y_test: pandas Series or DataFrame test target.
        :param cv: cross validation splits. By default, cv=None.
        :param cv_weights: array, weight for each split. By default, cv_weights=None.
        :param best_model_full_fit: boolean , whether to fit the best model to the whole data frame (train + test).
        By default, set to False.
        :param n_jobs: number of jobs to run in parallel. By default, n_jobs=1. If you are using XGBoost, n_jobs should
        stay equal to 1.

        :return: AutoMLResults Object
        """
        return self._run(x_train, x_test, y_train=y_train, y_test=y_test, cv=cv, cv_weights=cv_weights, n_jobs=n_jobs)


class AutoMLSpark(_AutoMLWrapper):
    def __init__(self, estimator, settings=AutoMLSettings(), models=None, fixed_params=None, hyper_params=None,
                 model_classes=None, feature_selection=None, visualizations=None):
        """
        This algorithm enables data scientists to find the best models among different options with different
        parametrization. It will pick the best model among the models, including the hyper params combination
        that performed the configured scoring the best.

        :param settings: AutoMLSettings
        :param estimator: model family to use: 'linear', 'binary', 'multi_class'.
        :param models: assets available in the functionality depending on estimator_family:

        linear:
            'LinearRegression' 'Lasso', 'Ridge', 'GBTRegressor'.

        binary:
            'LogisticRegression', 'SVC , 'RandomForestClassifier', 'GBTClassifier'.

        multi_class:
            'LogisticRegression' , 'RandomForestClassifier', 'GBTClassifier'

        :param fixed_params: dictionary, initialize each model with these fixed params. Each key should correspond to a
        key from the assets and values should be a dictionary: {'param_name': 'param_value'}. By default, the assets
        will be run with default configuration. Please, refer to online documentation (pyspark ..) to find out what are
        the possible parameters and values.

        :param hyper_params: dictionary of hyper_params for the GridSearch explorations. Each key should correspond to a
        key from the assets and values should be a dictionary: {'param_name': [list_of_values_to_explore]}. By default,
        the models will be run without any hyper_params. Please, refer to online documentation (pyspark ..) to find out
        what are the possible parameters and values.

        :param model_classes: dictionary of model classes to pass to be run by the AutoML.

        :param visualizations: dictionary where key is the name and value is the related function.
        Please refer to utils.visualizations to check which function are available.
        """
        super(AutoMLSpark, self).__init__(estimator, settings, models, fixed_params, hyper_params, model_classes,
                                          feature_selection, visualizations)

        if not _AutoMLSpark:
            raise AutoMLException("Error: install the `pyspark` library in order to use the AutoMLSpark.")

        self.auto_ml_class = _AutoMLSpark

    def run(self, train_data, test_data, variable_to_predict, cv=3) -> 'AutoMLResults':
        """
        Runs the AutoML

        :param train_data: pandas DataFrame of train set of features
        :param test_data: pandas DataFrame of test set of features
        :param variable_to_predict: column name of the variable to predict
        :param cv: cross validation splits. By default, cv=None.

        :return: AutoMLResults Object
        """
        return self._run(train_data, test_data, variable_to_predict=variable_to_predict, cv=cv)

