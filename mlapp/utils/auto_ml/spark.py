import pandas as pd
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, GBTClassifier
from pyspark.sql.dataframe import DataFrame
from pyspark.rdd import RDD
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from multiprocessing.pool import ThreadPool
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel
import pyspark.sql.functions as F
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorSlicer, VectorAssembler, ChiSqSelector, MinMaxScaler
import numpy as np
from mlapp.utils.exceptions.framework_exceptions import AutoMLException
from mlapp.utils.auto_ml.base import _AutoMLBase
from mlapp.utils.metrics.spark import classification, regression
from mlapp.utils.metrics.spark import MapeEvaluator, F1ScoreEvaluator
from mlapp.utils.visualizations.spark import visualization_methods


class CrossValidationSpark(CrossValidator):
    """
    An implementation of spark cross validator that can accept various train/test indices methods (under development)
    """
    def __init__(self, estimator=None, estimatorParamMaps=None, evaluator=None, cv=3, seed=None, parallelism=1,
                 collectSubModels=False):
        # check if cv is a list or a tuple containing indices. This is a preparation for time series cross validation.
        if isinstance(cv, tuple):
            self.sequentialIndex = True
            numFolds = 1
        elif isinstance(cv, list):
            self.sequentialIndex = True
            numFolds = len(cv)
        else:
            self.sequentialIndex = False
            numFolds = cv

        super(CrossValidationSpark, self).__init__(estimator=estimator, estimatorParamMaps=estimatorParamMaps, evaluator=evaluator, numFolds=numFolds,
                 seed=seed, parallelism=parallelism, collectSubModels=collectSubModels)

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        h = 1.0 / nFolds
        randCol = self.uid + "_rand"
        df = dataset.select("*", F.rand(seed).alias(randCol))
        metrics = np.zeros((numModels, nFolds))

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        for i in range(nFolds):
            if self.sequentialIndex:
                pass
                # todo pass a column name to base the split on. make sure the split conforms to sklearn norms.
                # idx = [1,2,3,4]
                # training.where(~col("id").isin(idx)).show()
            else:
                validateLB = i * h
                validateUB = (i + 1) * h
                condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
                validation = df.filter(condition).cache()
                train = df.filter(~condition).cache()

            tasks = self._parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam)
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics[j,i] = metric
                if collectSubModelsParam:
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        avgMetrics  = np.mean(metrics,axis=1)

        if eva.isLargerBetter():
            bestIndex = np.argmax(avgMetrics)
        else:
            bestIndex = np.argmin(avgMetrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(CrossValidatorModel(bestModel, avgMetrics.tolist(), subModels)), metrics

    @staticmethod
    def _parallelFitTasks(est, train, eva, validation, epm, collectSubModel):
        modelIter = est.fitMultiple(train, epm)
        def singleTask():
            index, model = next(modelIter)
            metric = eva.evaluate(model.transform(validation, epm[index]))
            return index, metric, model if collectSubModel else None
        return [singleTask] * len(epm)


class _AutoMLSpark(_AutoMLBase):
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

    def _validate_input(self, estimator_family, train, test, *args, **kwargs):
        if estimator_family not in ['linear', 'binary', 'multi_class']:
            raise AutoMLException("ERROR: model family not supported.")

        for data in [train, test]:
            if not (isinstance(data, DataFrame) or isinstance(data, RDD)):
                raise AutoMLException("ERROR: data type should be PySpark RDD or DataFrame.")

            if kwargs['variable_to_predict'] not in data.columns:
                raise AutoMLException(f"ERROR: variable_to_predict '{kwargs['variable_to_predict']}' "
                                      f"column missing in data input.")

        # TODO: add more validations tests

    def _visualization_methods(self, method):
        return visualization_methods(method)

    def _default_model_classes(self, estimator_family):
        return {
            'linear': {
                'LinearRegression': LinearRegression,
                'Lasso': LinearRegression,  # fixed_params sets elasticNetParams = 1, which makes this a Lasso model.
                'Ridge': LinearRegression,  # fixed_params sets elasticNetParams = 0, which makes this a Ridge model.
                'GBTRegressor': GBTRegressor
            },
            'binary': {
                'LogisticRegression': LogisticRegression,
                'SVC': LinearSVC,
                'RandomForestClassifier': RandomForestClassifier,
                'GBTClassifier': GBTClassifier
            },
            'multi_class': {
                'LogisticRegression': LogisticRegression,
                'RandomForestClassifier': RandomForestClassifier,
                'GBTClassifier': GBTClassifier
            }
        }[estimator_family]

    def _default_fixed_params(self, estimator_family):
        return {
            'linear': {
                'Lasso': {
                    'elasticNetParam': 1
                },
                'Ridge': {
                    'elasticNetParam': 0
                }
            }
            # TODO: add default fixed params
        }.get(estimator_family, {})

    def _default_hyper_params(self, estimator_family):
        return {
            # TODO: add default hyper params
        }.get(estimator_family, {})

    def _get_model_intercept_and_coefficients(self, model, features):
        try:
            return model.intercept, [list(c) for c in zip(features, list(model.coefficients))],
        except:
            try:
                return 0, [[features[model.featureImportances.indices[index]], value] for index, value in enumerate(model.featureImportances.values)]
            except:
                return 0, []

    def _calc_cv_df(self, model_key, metrics, param_map):
        new_param_map = []
        for d in param_map:
            new_dict = {}
            for k, v in d.items():
                new_dict[k.name] = v
            new_param_map.append(new_dict)
        res = pd.DataFrame(new_param_map)
        res.insert(0, 'std', np.std(metrics, axis=1))
        res.insert(0, 'mean_score', np.mean(metrics, axis=1))
        res.insert(0, 'estimator', model_key)
        return res

    def _get_cv_results(self, estimator_family, searches, features, scoring, greater_is_better, *args, **kwargs):
        keys = list(searches.keys())
        is_larger_better = self._get_scorer(estimator_family, scoring, *args, **kwargs).isLargerBetter()
        if is_larger_better:
            best_model_key = keys[int(np.argmax([max(searches[k][0].avgMetrics) for k in keys]))]
        else:
            best_model_key = keys[int(np.argmin([min(searches[k][0].avgMetrics) for k in keys]))]

        best_model = searches[best_model_key][0].bestModel
        best_cv_score = max(searches[best_model_key][0].avgMetrics)
        intercept, coefficients = self._get_model_intercept_and_coefficients(best_model, features)

        df = pd.concat([value[1] for value in searches.values()]).sort_values(
            by='mean_score', ascending=not greater_is_better)

        return df, best_model_key, best_model, best_cv_score, intercept, coefficients

    def _full_data_model_fit(self, *args, **kwargs):
        pass

    def _predict_train_test(self, model, train, test, *args, **kwargs):
        return model.transform(train), model.transform(test)

    def _feature_selection_globals(self, method):
        if method in globals():
            return globals()[method]

    def _run_feature_selection(self, feature_selection, train, test, *args, **kwargs):
        feature_selection_clf = self._load_feature_selection_method(feature_selection.get('method', ''))
        evaluated_params = self._eval_config_input(feature_selection.get('params', {}))

        features = [c for c in train.columns if c != kwargs['variable_to_predict']]
        assembler = VectorAssembler(inputCols=features, outputCol='features')
        v_train = assembler.transform(train)
        v_test = assembler.transform(test)

        v_train = v_train.select(['features', kwargs['variable_to_predict']])
        v_test = v_test.select(['features', kwargs['variable_to_predict']])

        if not feature_selection_clf:
            return features, v_train, v_test

        selector = feature_selection_clf(
            featuresCol='features', outputCol="selectedFeatures", **evaluated_params).fit(v_train)
        v_train = selector.transform(v_train).drop('features').withColumnRenamed("selectedFeatures", "features")
        v_test = selector.transform(v_test).drop('features').withColumnRenamed("selectedFeatures", "features")

        return [features[i] for i in selector.selectedFeatures], v_train, v_test

    def _get_scorer(self, estimator_family, metric=None, *args, **kwargs):
        variable_to_predict = kwargs['variable_to_predict']
        return {
            'areaUnderROC': BinaryClassificationEvaluator(metricName='areaUnderROC', labelCol=variable_to_predict),
            'areaUnderPR': BinaryClassificationEvaluator(metricName='areaUnderPR', labelCol=variable_to_predict),
            'f1_score': F1ScoreEvaluator(labelCol=variable_to_predict),
            'rmse': RegressionEvaluator(metricName='rmse', labelCol=variable_to_predict),
            'mse': RegressionEvaluator(metricName='mse', labelCol=variable_to_predict),
            'r2': RegressionEvaluator(metricName='r2', labelCol=variable_to_predict),
            'mae': RegressionEvaluator(metricName='mae', labelCol=variable_to_predict),
            'mape': MapeEvaluator(labelCol=variable_to_predict)
        }.get(metric, {
                'linear': RegressionEvaluator(metricName='mse', labelCol=variable_to_predict),
                'non_linear': RegressionEvaluator(metricName='mse', labelCol=variable_to_predict),
                'multi_class': F1ScoreEvaluator(labelCol=variable_to_predict),
                'binary': F1ScoreEvaluator(labelCol=variable_to_predict),
            }.get(estimator_family))

    def _scores_function(self, scores_summary_func, estimator_family):
        # take scores summary by 'scores_summary' then by 'best_estimator'
        return {
            'classification': classification,
            'regression': regression,
        }.get(scores_summary_func, {
            'linear': regression,
            'non_linear': regression,
            'multi_class': classification,
            'binary': classification
        }.get(estimator_family))

    def _inner_search(self, estimator_family, train, test, model_key, model_class, fixed_params, hyper_params, scoring,
                      greater_is_better, cv, *args, **kwargs):
        model_object = model_class(labelCol=kwargs['variable_to_predict'], **fixed_params.get(model_key, {}))
        tuned_parameters = hyper_params.get(model_key, {})

        # build hyper parameter grid:
        pgb = ParamGridBuilder()
        for tp_key in tuned_parameters:
            pgb = pgb.addGrid(getattr(model_object, tp_key), tuned_parameters[tp_key])
        param_map = pgb.build()

        # run cross validator:
        evaluator = self._get_scorer(estimator_family, scoring, *args, **kwargs)
        cv_f = CrossValidationSpark(
            estimator=model_object, estimatorParamMaps=param_map, evaluator=evaluator, cv=cv)

        hps, metrics = cv_f.fit(train)

        return hps, self._calc_cv_df(model_key, metrics, param_map)



