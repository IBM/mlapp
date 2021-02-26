<a name="mlapp.utils.metrics.spark"></a>
# mlapp.utils.metrics.spark

<a name="mlapp.utils.metrics.spark.classification"></a>
#### classification

```python
classification(train_predictions, test_predictions, train_actuals=None, test_actuals=None, variable_to_predict='target', prediction_col='prediction', *args, **kwargs)
```

Return a dictionary of accuracy scores for provided predicted values. The following scores are returned:
Training Precision, Training Recall, Training f1_score, Training areaUnderROC, Training areaUnderPR,
Testing Precision, Testing Recall, Testing f1_score, Testing areaUnderROC, Testing areaUnderPR

**Arguments**:

- `train_predictions`: train predicted values.
- `test_predictions`: test actual values.
- `variable_to_predict`: variable_to_predict.
- `prediction_col`: predictionCol

**Returns**:

dictionary with accuracy scores.

<a name="mlapp.utils.metrics.spark.regression"></a>
#### regression

```python
regression(train_predictions, test_predictions, train_actuals=None, test_actuals=None, variable_to_predict='target', prediction_col='prediction', *args, **kwargs)
```

Return a dictionary of accuracy scores for provided predicted values. The following scores are returned:
Training Accuracy(R^2), Testing Accuracy(R^2), Training MAPE, testing MAPE.

**Arguments**:

- `train_predictions`: train predicted values.
- `test_predictions`: test actual values.
- `variable_to_predict`: variable_to_predict.
- `prediction_col`: predictionCol.

**Returns**:

dictionary with accuracy scores.

<a name="mlapp.utils.metrics.spark.MapeEvaluator"></a>
## MapeEvaluator Objects

```python
class MapeEvaluator(Evaluator)
```

A mean absolute percentage error evaluator for spark ML

<a name="mlapp.utils.metrics.spark.F1ScoreEvaluator"></a>
## F1ScoreEvaluator Objects

```python
class F1ScoreEvaluator(Evaluator)
```

An F1 evaluator for spark ML

