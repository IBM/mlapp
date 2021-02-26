import inspect
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from mlapp.utils.exceptions.framework_exceptions import AutoMLException


class _AutoMLBase(ABC):
    @abstractmethod
    def _validate_input(self, estimator_family, train, test, *args, **kwargs):
        pass

    @abstractmethod
    def _default_fixed_params(self, estimator_family):
        pass

    @abstractmethod
    def _default_hyper_params(self, estimator_family):
        pass

    @abstractmethod
    def _default_model_classes(self, estimator_family):
        pass

    def _get_models_names(self, estimator_family, models=None):
        if models:
            return models
        else:
            return list(self._default_model_classes(estimator_family).keys())

    @abstractmethod
    def _eval_config_input(self, config_input):
        pass

    def construct_dictionary(self, input_dict, default_dict):
        copy_dict = deepcopy(default_dict)
        if input_dict and isinstance(input_dict, dict):
            copy_dict.update(self._eval_config_input(input_dict))

        return copy_dict

    @abstractmethod
    def _get_model_intercept_and_coefficients(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_cv_results(self, estimator_family, searches, features, scoring, greater_is_better, *args, **kwargs):
        pass

    @abstractmethod
    def _inner_search(self, estimator_family, train, test, model_key, model_class, fixed_params, hyper_params, cv,
                      scoring, greater_is_better, *args, **kwargs):
        pass

    @abstractmethod
    def _visualization_methods(self, method):
        pass

    def _create_images(self, methods, *args, **kwargs):
        figs = {}
        for method in methods:
            vis_f = self._visualization_methods(method)
            if not vis_f:
                print("INFO: Visualization method '{0}' doesn't exist ".format(method))
                continue
            signature = inspect.signature(vis_f)
            f_args = list(signature.parameters)
            parameters = {k: v.default for k, v in signature.parameters.items() if
                          v.default is not inspect.Parameter.empty}
            passed_parameters = {arg: kwargs[arg] for arg in f_args if arg in kwargs}
            parameters.update(passed_parameters)

            missing_params = set(f_args) - set(parameters.keys())
            if len(missing_params) > 0:
                print(
                    "INFO: Failed to create visualization {0}. Missing Params: {1}".format(method, str(missing_params)))
            else:
                figure = vis_f(*[parameters[arg] for arg in f_args])
                if figure is not None:
                    figs[method] = figure
        return figs

    @abstractmethod
    def _full_data_model_fit(self, model, train, test, *args, **kwargs):
        pass

    @abstractmethod
    def _predict_train_test(self, model, train, test, *args, **kwargs):
        pass

    @abstractmethod
    def _get_scorer(self, estimator_family, metric=None, *args, **kwargs):
        pass

    def _construct_run_params(self, estimator_family, fixed_params, hyper_params, model_classes):
        return self.construct_dictionary(fixed_params, self._default_fixed_params(estimator_family)), \
               self.construct_dictionary(hyper_params, self._default_hyper_params(estimator_family)), \
               self.construct_dictionary(model_classes, self._default_model_classes(estimator_family))

    def _load_feature_selection_method(self, method):
        if callable(method):
            return method
        elif isinstance(method, str) and method != 'AllFeatures':
            globals_method = self._feature_selection_globals(method)
            if not globals_method:
                print(f"INFO: feature selection method '{method}' not in scope.")
            return globals_method

    @abstractmethod
    def _feature_selection_globals(self, method):
        pass

    @abstractmethod
    def _run_feature_selection(self, feature_selection, train, test, *args, **kwargs):
        pass

    @staticmethod
    def _get_feature_selection_methods(feature_selection):
        return feature_selection if feature_selection else [{"method": "AllFeatures"}]

    @staticmethod
    def _get_feature_selection_name(method):
        if isinstance(method, str):
            return method
        else:
            return method.__class__.__name__

    @abstractmethod
    def _scores_function(self, scores_summary_function, estimator_family):
        pass

    def run(self, results, scores_summary, feature_selections, estimator_family, train, test, models=None,
            fixed_params=None, hyper_params=None, model_classes=None, scoring=None, greater_is_better=True, *args,
            **kwargs):

        # validate input
        self._validate_input(estimator_family, train, test, *args, **kwargs)
        # init params
        fixed_params, hyper_params, model_classes = \
            self._construct_run_params(estimator_family, fixed_params, hyper_params, model_classes)

        for feature_selection_method in self._get_feature_selection_methods(feature_selections):
            # feature selection
            selected_features, train_filtered, test_filtered = \
                self._run_feature_selection(feature_selection_method, train, test, *args, **kwargs)

            # run hyper param search per model
            hyper_params_searches = {}
            for model_key in self._get_models_names(estimator_family, models):
                try:
                    model_class = model_classes.get(model_key)
                    if not model_class:
                        print(f"WARNING: `{model_key}` is not supported by MLApp's AutoML. "
                              f"Check naming is correct and that there is no missing python library installation.")
                        continue

                    start_time = time.time()

                    hyper_params_searches[model_key] = self._inner_search(
                        estimator_family, train_filtered, test_filtered, model_key, model_class, fixed_params,
                        hyper_params, scoring, greater_is_better, *args, **kwargs)

                    print("INFO: hyper params search took %.2f seconds for model: %s parameter settings." %
                          ((time.time() - start_time), model_key))

                except Exception as err:
                    raise AutoMLException(
                        f"GridSearch/RandomSearch failed with '{model_key}' model with error: {str(err)}.")

            if len(hyper_params_searches.keys()) == 0:
                raise AutoMLException(f"AutoML has no output! "
                                      f"Check if you are running at least one model or whether all models failed.")

            # get cv results
            cv_results, best_model_key, best_model, best_cv_score, intercept, coefficients = \
                self._get_cv_results(
                    estimator_family, hyper_params_searches, selected_features, scoring, greater_is_better, *args,
                    **kwargs)

            # add figures
            output_plots = {}
            if kwargs.get('visualizations'):
                output_plots = self._create_images(
                    kwargs['visualizations'], hyper_params=hyper_params.get(best_model_key, {}),
                    grid_search_results=hyper_params_searches[best_model_key], greater_is_better=greater_is_better,
                    X_train=train_filtered, X_test=test_filtered, model=best_model, *args, **kwargs)

            # get train/test y_hat
            train_y_hat, test_y_hat = self._predict_train_test(
                best_model, train_filtered, test_filtered, *args, **kwargs)

            # fit model on full train+test data
            if kwargs.get('best_model_full_fit'):
                best_model = self._full_data_model_fit(best_model, train, test, *args, **kwargs)

            if best_model is None:
                raise AutoMLException("Error: AutoML run must pass 'models' with at least one model!")

            # append results
            scores_function = self._scores_function(scores_summary, estimator_family)
            results.add_cv_run(scores_function, estimator_family,
                               self._get_feature_selection_name(feature_selection_method.get('method')), train_y_hat,
                               test_y_hat, selected_features, best_model, intercept, coefficients, best_cv_score,
                               cv_results, output_plots, *args, **kwargs)

        # return results
        return results
