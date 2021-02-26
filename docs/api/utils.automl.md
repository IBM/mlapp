<a name="mlapp.utils.automl"></a>
# mlapp.utils.automl

<a name="mlapp.utils.automl.AutoMLSettings"></a>
## AutoMLSettings

```python
class AutoMLSettings(metric=None, greater_is_better=True, search_type='grid', scores_summary=None)
```

**Arguments**:

- `metric`: scorer function for cross validation. By default auto-selects based on estimator.
>`linear`: mean squared error
>
>`non_linear`: mean squared error
>
>`binary`: f1 score
>
>`multi_class`: f1 score
> 
> Accepts scorer objects depending on framework. e.g `r2_score` from `sklearn.metrics` when using AutoMLPandas. 
- `greater_is_better`: boolean, i.e. for MAPE, low is better.
- `search_type`: 'grid'/'random' - type of search to be run to find the best combination.
- `score_summary`: 'regression'/'classification'/'time_series' or None auto-selects by estimator.

<a name="mlapp.utils.automl.AutoMLPandas"></a>
## AutoMLPandas

```python
class AutoMLPandas(estimator, settings=AutoMLSettings(), models=None, fixed_params=None, hyper_params=None, model_classes=None, feature_selection=None, visualizations=None)
```

This algorithm enables data scientists to find the best models among different options with different
parametrization. It will pick the best model among the models, including the hyper params combination
that performed the configured scoring the best.

**Arguments**:

- `settings`: AutoMLSettings
- `estimator`: model family to use: 'linear', 'non_linear', 'binary', 'multi_class'.
- `models`: assets available in the functionality depending on estimator_family:
> `linear`:
> 'Lasso' 'LinearRegression', 'Ridge', 'ElasticNet', 'Lars', 'LassoLars' and 'BayesianRidge'.
> 
> `non_linear`:
> 'RandomForest', 'ExtraTree', 'XGBoost', 'LightGBM' and 'GradientBoosting'.
> 
> `binary`:
> 'Logistic', 'RBF , 'MultiLogistic', 'ExtraTreeClassifier', 'RandomForest', 'GaussianNB',
> 'GaussianProcess_Classifier', 'KnearestNeighbors', 'GradientBoosting', 'XGBoostClassifier',
> 'LightGBMClassifier', 'GradientBoostingClassifier' and 'AdaBoostClassifier'.
> 
> `multi_class`:
> 'Logistic', 'RBF , 'MultiLogistic', 'ExtraTreeClassifier', 'RandomForest', 'GaussianNB',
> 'GaussianProcess_Classifier', 'KnearestNeighbors', 'GradientBoosting', 'XGBoostClassifier',
> 'LightGBMClassifier', 'GradientBoostingClassifier' and 'AdaBoostClassifier'.
> 
> `clustering`:
> 'AgglomerativeClustering', 'KMeans', 'MiniBatchKmeans', 'DB-SCAN', 'MeanShift' and 'SpectralClustering'.
- `fixed_params`: dictionary, initialize each model with these fixed params. Each key should correspond to a key
from the assets and values should be a dictionary: {'param_name': 'param_value'}. By default, the assets will be
run with default configuration. Please, refer to online documentation (sklearn ..) to find out what are the possible
parameters and values.

- `hyper_params`: dictionary of hyper_params for the GridSearch explorations. Each key should correspond to a key
from the assets and values should be a dictionary: {'param_name': [list_of_values_to_explore]}. By default, the
assets will be run without any hyper_params. Please, refer to online documentation (sklearn ..) to find out
what are the possible parameters and values.

- `model_classes`: dictionary of model classes to pass to be run by the AutoML.

- `visualizations`: dictionary where key is the name and value is the related function.
Please refer to utils.visualizations to check which function are available.

<a name="mlapp.utils.automl.AutoMLPandas.run"></a>
#### run

```python
run(x_train, y_train, x_test, y_test, cv=5, cv_weights=None, n_jobs=1, best_model_full_fit=False) -> 'AutoMLResults'
```

Runs the AutoML

**Arguments**:

- `x_train`: pandas DataFrame of train set of features
- `y_train`: pandas Series or DataFrame train target.
- `x_test`: pandas DataFrame of test set of features
- `y_test`: pandas Series or DataFrame test target.
- `cv`: cross validation splits. By default, cv=None.
- `cv_weights`: array, weight for each split. By default, cv_weights=None.
- `best_model_full_fit`: boolean , whether to fit the best model to the whole data frame (train + test).
By default, set to False.
- `n_jobs`: number of jobs to run in parallel. By default, n_jobs=1. If you are using XGBoost, n_jobs should
stay equal to 1.

**Returns**:

AutoMLResults Object

<a name="mlapp.utils.automl.AutoMLSpark"></a>
## AutoMLSpark

```python
class AutoMLSpark(estimator, settings=AutoMLSettings(), models=None, fixed_params=None, hyper_params=None, model_classes=None, feature_selection=None, visualizations=None)
```

This algorithm enables data scientists to find the best models among different options with different
parametrization. It will pick the best model among the models, including the hyper params combination
that performed the configured scoring the best.

**Arguments**:

- `settings`: AutoMLSettings
- `estimator`: model family to use: 'linear', 'binary', 'multi_class'.
- `models`: assets available in the functionality depending on estimator_family:
> `linear`:
> 'LinearRegression' 'Lasso', 'Ridge', 'GBTRegressor'.
> 
> `binary`:
> 'LogisticRegression', 'SVC , 'RandomForestClassifier', 'GBTClassifier'.
> 
> `multi_class`:
> 'LogisticRegression' , 'RandomForestClassifier', 'GBTClassifier'
- `fixed_params`: dictionary, initialize each model with these fixed params. Each key should correspond to a
key from the assets and values should be a dictionary: {'param_name': 'param_value'}. By default, the assets
will be run with default configuration. Please, refer to online documentation (pyspark ..) to find out what are
the possible parameters and values.

- `hyper_params`: dictionary of hyper_params for the GridSearch explorations. Each key should correspond to a
key from the assets and values should be a dictionary: {'param_name': [list_of_values_to_explore]}. By default,
the models will be run without any hyper_params. Please, refer to online documentation (pyspark ..) to find out
what are the possible parameters and values.

- `model_classes`: dictionary of model classes to pass to be run by the AutoML.

- `visualizations`: dictionary where key is the name and value is the related function.
Please refer to utils.visualizations to check which function are available.

<a name="mlapp.utils.automl.AutoMLSpark.run"></a>
#### run

```python
run(train_data, test_data, variable_to_predict, cv=3) -> 'AutoMLResults'
```

Runs the AutoML

**Arguments**:

- `train_data`: pandas DataFrame of train set of features
- `test_data`: pandas DataFrame of test set of features
- `variable_to_predict`: column name of the variable to predict
- `cv`: cross validation splits. By default, cv=None.

**Returns**:

AutoMLResults Object

<a name="mlapp.utils.automl.AutoMLResults"></a>
## AutoMLResults

```python
class AutoMLResults(settings=AutoMLSettings())
```

AutoMLResults Object contains results of AutoML run/s.

**Arguments**:

- `settings`: AutoMLSettings Object

<a name="mlapp.utils.automl.AutoMLResults.get_best_model"></a>
#### get\_best\_model

```python
get_best_model()
```

Gets best model from AutoML run.

<a name="mlapp.utils.automl.AutoMLResults.get_metrics"></a>
#### get\_metrics

```python
 get_metrics()
```

Gets metrics result associated with the best model.

<a name="mlapp.utils.automl.AutoMLResults.get_figures"></a>
#### get\_figures

```python
 get_figures()
```

Gets figures produced by AutoML run associated with the best model.

<a name="mlapp.utils.automl.AutoMLResults.print_report"></a>
#### print\_report

```python
print_report(full=True, ascending=False)
```

Prints score report of all runs done by the AutoML.

**Arguments**:

- `full`: prints scores from all model runs, or use just best representative from each model (True/False)
- `ascending`: show scores in ascending order (True/False)

<a name="mlapp.utils.automl.AutoMLResults.get_metadata"></a>
#### get\_metadata

```python
get_metadata()
```

Gets all metadata associated with the best model.

> scores: scores associated with estimator used
>
> estimator_family: estimator family (linear/non-linear/classificaton/etc.)
>
> model: best model class name
>
> selected_features_names: selected features
>
> intercept: intercept of model if exists
>
> coefficients: coefficients / feature importance if exist
>
> feature_selection: feature selection method
>
> cv_score: cross validation score

<a name="mlapp.utils.automl.AutoMLResults.get_train_predictions"></a>
#### get\_train\_predictions

```python
get_train_predictions()
```

Gets train predictions from the best model.

<a name="mlapp.utils.automl.AutoMLResults.get_test_predictions"></a>
#### get\_test\_predictions

```python
get_test_predictions()
```

Gets test predictions from the best model.

<a name="mlapp.utils.automl.AutoMLResults.predict_proba_by_threshold"></a>
#### predict\_proba\_by\_threshold

```python
predict_proba_by_threshold(x_train, x_test, y_train, y_test, threshold)
```

Prints score report of all runs done by the AutoML.

**Arguments**:

- `x_train`: train data (pd.DataFrame)
- `x_test`: test data (pd.DataFrame)
- `y_train`: train data (pd.DataFrame/pd.Series)
- `y_test`: test data (pd.DataFrame/pd.Series)
- `threshold`: Threshold percentage/s to use for predict by probability ([float]/float)

**Returns**:

dictionary containing relevant scores

