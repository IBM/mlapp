from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
import pyspark.sql.functions as F
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator, MulticlassClassificationEvaluator, \
    BinaryClassificationEvaluator


def classification(train_predictions, test_predictions, train_actuals=None, test_actuals=None,
                   variable_to_predict='target', prediction_col='prediction', *args, **kwargs):
    """
    Return a dictionary of accuracy scores for provided predicted values. The following scores are returned:
    Training Precision, Training Recall, Training f1_score, Training areaUnderROC, Training areaUnderPR,
    Testing Precision, Testing Recall, Testing f1_score, Testing areaUnderROC, Testing areaUnderPR
    :param train_predictions: train predicted values.
    :param test_predictions: test actual values.
    :param variable_to_predict: variable_to_predict.
    :param prediction_col: predictionCol
    :return: dictionary with accuracy scores.
    """
    evaluatorMulti = MulticlassClassificationEvaluator(labelCol=variable_to_predict, predictionCol=prediction_col)
    evaluatorRoc = BinaryClassificationEvaluator(
        labelCol=variable_to_predict, rawPredictionCol="prediction", metricName='areaUnderROC')
    evaluatorPR = BinaryClassificationEvaluator(
        labelCol=variable_to_predict, rawPredictionCol="prediction", metricName='areaUnderPR')

    labels = sorted(train_predictions.select(prediction_col).rdd.distinct().map(lambda r: r[0]).collect())
    train_metrics = train_predictions.select(variable_to_predict, prediction_col)
    test_metrics = test_predictions.select(variable_to_predict, prediction_col)

    scores = {}
    for label in labels:
        scores.update({
            'precision (' + str(label) + ') (train set)': evaluatorMulti.evaluate(
                train_metrics, {evaluatorMulti.metricName: "weightedPrecision"}),
            'recall (' + str(label) + ') (test set)': evaluatorMulti.evaluate(
                train_metrics, {evaluatorMulti.metricName: "weightedRecall"}),
            'f1 score (' + str(label) + ') (train set)': evaluatorMulti.evaluate(
                train_metrics, {evaluatorMulti.metricName: "f1"}),
            'accuracy (' + str(label) + ') (train set)': evaluatorMulti.evaluate(
                train_metrics, {evaluatorMulti.metricName: "accuracy"}),
            'areaUnderROC (' + str(label) + ') (train set)': evaluatorRoc.evaluate(train_metrics),
            'areaUnderPR (' + str(label) + ') (train set)': evaluatorPR.evaluate(train_metrics),
            'precision (' + str(label) + ') (test set)': evaluatorMulti.evaluate(
                test_metrics, {evaluatorMulti.metricName: "weightedPrecision"}),
            'recall (' + str(label) + ') (test set)': evaluatorMulti.evaluate(
                test_metrics, {evaluatorMulti.metricName: "weightedRecall"}),
            'f1_score (' + str(label) + ') (test set)': evaluatorMulti.evaluate(
                test_metrics, {evaluatorMulti.metricName: "f1"}),
            'accuracy (' + str(label) + ') (test set)': evaluatorMulti.evaluate(
                test_metrics, {evaluatorMulti.metricName: "accuracy"}),
            'areaUnderROC (' + str(label) + ') (test set)': evaluatorRoc.evaluate(test_metrics),
            'areaUnderPR (' + str(label) + ') (test set)': evaluatorPR.evaluate(test_metrics)
        })
    return scores


def regression(train_predictions, test_predictions, train_actuals=None, test_actuals=None,
               variable_to_predict='target', prediction_col='prediction', *args, **kwargs):
    """
    Return a dictionary of accuracy scores for provided predicted values. The following scores are returned:
    Training Accuracy(R^2), Testing Accuracy(R^2), Training MAPE, testing MAPE.
    :param train_predictions: train predicted values.
    :param test_predictions: test actual values.
    :param variable_to_predict: variable_to_predict.
    :param prediction_col: predictionCol.
    :return: dictionary with accuracy scores.
    """
    reg_eval = RegressionEvaluator(labelCol=variable_to_predict, predictionCol=prediction_col)
    mape_eval = MapeEvaluator(labelCol=variable_to_predict, predictionCol=prediction_col)
    return {
        'R2 (train set)': reg_eval.evaluate(train_predictions, {reg_eval.metricName: 'r2'}),
        'R2 (test set)': reg_eval.evaluate(test_predictions, {reg_eval.metricName: 'r2'}),
        'MAE (train set)': reg_eval.evaluate(train_predictions, {reg_eval.metricName: 'mae'}),
        'MAE (test set)': reg_eval.evaluate(test_predictions, {reg_eval.metricName: 'mae'}),
        'RMSE (train set)': reg_eval.evaluate(train_predictions, {reg_eval.metricName: 'rmse'}),
        'RMSE (test set)': reg_eval.evaluate(test_predictions, {reg_eval.metricName: 'rmse'}),
        'MAPE (train set)': mape_eval.evaluate(train_predictions),
        'MAPE (test set)': mape_eval.evaluate(test_predictions)
    }


class MapeEvaluator(Evaluator):
    """
    A mean absolute percentage error evaluator for spark ML
    """
    def __init__(self, predictionCol="prediction", labelCol="label"):
        super(MapeEvaluator, self).__init__()
        self.predictionCol = predictionCol
        self.labelCol = labelCol

    def _evaluate(self, dataset):
        dataset = dataset.withColumn('non_zero',
                                     F.when(F.col(self.predictionCol) == 0, 1).otherwise(F.col(self.predictionCol)))
        return (dataset.select(F.mean(
                F.abs(
                    (F.col(self.labelCol) - F.col(self.predictionCol)) / F.col('non_zero'))).alias('mape')) \
                .collect()[0][0]) * float(100)

    def isLargerBetter(self):
        return False


class F1ScoreEvaluator(Evaluator):
    """
    An F1 evaluator for spark ML
    """
    def __init__(self, predictionCol="prediction", labelCol="label"):
        super(F1ScoreEvaluator, self).__init__()
        self.predictionCol = predictionCol
        self.labelCol = labelCol

    def _evaluate(self, dataset):
        eval_multi = MulticlassClassificationEvaluator(labelCol=self.labelCol, predictionCol=self.predictionCol)
        return eval_multi.evaluate(dataset.select(self.labelCol, self.predictionCol), {eval_multi.metricName: "f1"})

    def isLargerBetter(self):
        return True

