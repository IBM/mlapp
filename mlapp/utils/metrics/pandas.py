import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, \
    fbeta_score, r2_score, mean_absolute_error, mean_squared_error


def regression(train_y_predict, test_y_predict, train_y_real, test_y_real, *args, **kwargs):
    """
    Return a dictionary of accuracy scores for provided predicted values.
    The following scores are returned: Training Accuracy(R^2), Testing Accuracy(R^2), Training MAPE, Testing MAPE.
    :param train_y_predict: y train predicted values.
    :param train_y_real: y train actual values.
    :param test_y_predict: y test predicted values.
    :param test_y_real: y test actual values.
    :return: dictionary with accuracy scores.
    """
    try:
        train_y_predict_ravel = np.ravel(train_y_predict)
        test_y_predict_ravel = np.ravel(test_y_predict)
        train_y_actuals_ravel = np.ravel(train_y_real)
        test_y_actuals_ravel = np.ravel(test_y_real)
    except Exception as err:
        raise ("Cannot run `np.ravel` on one of the inputs: " + str(err))

    return {
        'R2 (train set)': r2_score(train_y_actuals_ravel, train_y_predict_ravel),
        'R2 (test set)': r2_score(test_y_actuals_ravel, test_y_predict_ravel),
        'MAPE (train set)': _get_mean_absolute_percentage_error(train_y_predict_ravel, train_y_actuals_ravel),
        'MAPE (test set)': _get_mean_absolute_percentage_error(test_y_predict_ravel, test_y_actuals_ravel),
        'MAE (train set)': mean_absolute_error(train_y_actuals_ravel, train_y_predict_ravel),
        'MAE (test set)': mean_absolute_error(test_y_actuals_ravel, test_y_predict_ravel),
        'RMSE (train set)': np.sqrt(mean_squared_error(train_y_predict_ravel, train_y_actuals_ravel)),
        'RMSE (test set)': np.sqrt(mean_squared_error(test_y_predict_ravel, test_y_actuals_ravel))
    }


def classification(train_y_predict, test_y_predict, train_y_real, test_y_real, beta=1, unbalanced=False, binary=None,
                   *args, **kwargs):
    """
    Return a dictionary of accuracy scores for provided predicted values. The following scores are returned:
    Jaccard Score Training, Jaccard Score Testing, AUC Score Training, AUC Score Testing, F_Beta Training Accuracy,
    F_Beta Testing Accuracy, recall Training Accuracy, recall Testing Accuracy, precision Training Accuracy,
    precision Testing Accuracy
    :param train_y_predict: y train predicted values.
    :param train_y_real: y train actual values.
    :param test_y_predict: y test predicted values.
    :param test_y_real: y test actual values.
    :param beta: F beta parameter. If beta=1, f1-score will be calculated
    :param unbalanced: Boolean. For now, set unbalanced=False --> to be fixed.
    :param binary: Boolean. If binary=True, average='binary' for F-beta calculation, else average='weighted'.
    Please refer to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html to understand the
    meaning
    :return: dictionary with accuracy scores.
    """
    if kwargs.get('test_y_proba') is not None and kwargs.get('train_y_proba') is not None:
        try:
            training_auc = float(roc_auc_score(train_y_real, kwargs['train_y_proba']))
            testing_auc = float(roc_auc_score(test_y_real, kwargs['test_y_proba']))
        except:
            print('INFO: scores summary no AUC accuracy')
            training_auc = None
            testing_auc = None
    else:
        training_auc = None
        testing_auc = None
    try:
        training_accuracy = float(accuracy_score(train_y_real, train_y_predict))
        testing_accuracy = float(accuracy_score(test_y_real, test_y_predict))
    except:
        print('INFO: Scores summary no accuracy')
        training_accuracy = None
        testing_accuracy = None

    if unbalanced:
        try:
            fbeta_score_train = _unbalanced_data(train_y_real, train_y_predict, beta=beta,
                                                 return_value=['precision', 'recall', 'F_beta'])
            fbeta_score_test = _unbalanced_data(test_y_real, test_y_predict, beta=beta,
                                                return_value=['precision', 'recall', 'F_beta'])
        except Exception as e:
            print(e)
            print('No F1-scoring for unbalanced')
            fbeta_score_train = None
            fbeta_score_test = None
    else:
        try:
            if binary is True:
                average = 'binary'
            elif binary is False:
                average = 'weighted'
            else:
                average = 'weighted' if len(pd.unique(train_y_real)) > 2 else 'binary'

            fbeta_score_train = {'recall': recall_score(train_y_real, train_y_predict, average=average),
                                 'precision': precision_score(train_y_real, train_y_predict, average=average),
                                 'F_beta': fbeta_score(train_y_real, train_y_predict, beta=beta, average=average)}
            fbeta_score_test = {'recall': recall_score(test_y_real, test_y_predict, average=average),
                                'precision': precision_score(test_y_real, test_y_predict, average=average),
                                'F_beta': fbeta_score(test_y_real, test_y_predict, beta=beta, average=average)}
        except Exception as e:
            print(e)
            print('No F1-scoring for unbalanced')
            fbeta_score_train = None
            fbeta_score_test = None

    scores = {}
    if training_accuracy is not None and testing_accuracy is not None:
        scores['jaccard (train set)'] = training_accuracy
        scores['jaccard (test set)'] = testing_accuracy
    if training_auc is not None and testing_auc is not None:
        scores['AUC (train set)'] = training_auc
        scores['AUC (test set)'] = testing_auc
    if fbeta_score_train is not None and fbeta_score_test is not None:
        scores['f1_score (train set)'] = fbeta_score_train.get('F_beta')
        scores['f1_score (test set)'] = fbeta_score_test.get('F_beta')
        scores['recall (train set)'] = fbeta_score_train.get('recall')
        scores['recall (test set)'] = fbeta_score_test.get('recall')
        scores['precision (train set)'] = fbeta_score_train.get('precision')
        scores['precision (test set)'] = fbeta_score_test.get('precision')

    return scores


def time_series(train_y_predict, test_y_predict, train_y_real, test_y_real, *args, **kwargs):
    """
    Return a dictionary of accuracy scores for provided predicted values, specifically in case of time series challenges.
    The following scores are returned: Training Accuracy(R^2), Testing Accuracy(R^2), Training MAPE, Testing MAPE,
    Training MASE, Testing MASE.
    :param train_y_predict: y train predicted values.
    :param train_y_real: y train actual values.
    :param test_y_predict: y test predicted values.
    :param test_y_real: y test actual values.
    :return: dictionary with accuracy scores.
    """
    return {'R2 (train set)': r2_score(train_y_real, train_y_predict),
            'R2 (test set)': r2_score(test_y_real, test_y_predict),
            'MAPE (train set)': _get_mean_absolute_percentage_error(train_y_predict, train_y_real),
            'MAPE (test set)': _get_mean_absolute_percentage_error(test_y_predict, test_y_real),
            'MASE (train set)': _get_mean_average_scaled_error(train_y_predict, train_y_real),
            'MASE (test set)': _get_mean_average_scaled_error(test_y_predict, test_y_real)}


def _get_mean_absolute_percentage_error(y_predictions, y_actuals):
    '''
    Mean absolute percentage error (MAPE) calculations
    :param y_predictions: labels estimations
    :param y_actuals: true labels
    :return: MAPE (float)
    '''
    y_actuals[np.where(y_actuals == 0)] = 1
    return np.mean(np.abs((y_actuals - y_predictions) / y_actuals)) * 100


def _get_mean_average_scaled_error(y_predictions, y_actuals, peak_dates=None):
    '''
    Mean average scaled error (MASE) calculations
    :param y_predictions: labels estimations
    :param y_actuals: true labels
    :param peak_dates: list of peak dates to exclude from calculations
    :return: MASE (float)
    '''
    y_a = y_actuals.copy()
    y_p = y_predictions.copy()
    if peak_dates is None:
        naive_model_error = np.mean(np.abs(y_a[1:] - np.roll(y_a, 1)[1:]))
    else:
        # remove zero index if exists because can't "naive" predict it
        peak_dates_no_zero_index = peak_dates.copy()
        zero_index = np.argwhere(peak_dates == 0)
        if len(zero_index) > 0:
            peak_dates_no_zero_index = np.delete(peak_dates_no_zero_index, zero_index)
        naive_model_error = np.mean(
            np.abs(y_actuals[peak_dates_no_zero_index] - y_actuals[peak_dates_no_zero_index - 1]))
        y_a = y_actuals[peak_dates_no_zero_index]
        y_p = y_predictions[peak_dates_no_zero_index]

    # MASE
    if naive_model_error != 0:
        mase = np.mean(np.abs(y_a - y_p) / naive_model_error)
    # MAD/Mean Ratio (when naive_model_error is zero)
    else:
        y_actuals_mean = np.mean(y_a)
        if y_actuals_mean == 0:
            y_actuals_mean = y_actuals_mean + 1
        mase = np.mean(np.abs(y_a - y_p) / y_actuals_mean)
    return mase


def _unbalanced_data(y_true, y_pred, beta=1, negative_label=None, return_value=None):
    """
    Returns scores in case of unbalanced data (all categories in target columns are not represented the same).
    This function is meant to overweight low-represented categories (or anomalies) in order to emphasize the ability of
    the model to detect specific cluster.
    :param y_true: true labels
    :param y_pred: estimated labels
    :param beta: beta of the F-beta calculatation. By default, beta=1 (F1-score)
    :param negative_label: List of labels to unit for the negative labels. The rest will be considered as positive.
    :param return_value: List of scores to return: 'F_beta', 'precision', or 'recall'
    :return: Dictionary with all the scores.
    """
    if negative_label is None:
        negative_label = [0]
    if return_value is None:
        return_value = ['F_beta']

    if isinstance(y_true, pd.DataFrame):
        y_true_reshaped = y_true.iloc[:, 0]
    elif not isinstance(y_true, pd.Series):
        y_true_reshaped = pd.Series(y_true)
    else:
        y_true_reshaped = y_true.copy()

    if isinstance(y_pred, pd.DataFrame):
        y_pred_reshaped = y_pred.iloc[:, 0]
    elif not isinstance(y_pred, pd.Series):
        y_pred_reshaped = pd.Series(y_pred, index=y_true.index)
    else:
        y_pred_reshaped = y_pred.copy()

    # Get the positive and the negative labels
    positive_labels = list(set(y_true_reshaped.values) - set(negative_label))

    # Replace negative_labels by 0 and positive labels by 1
    y_true_reshaped = y_true_reshaped.replace(negative_label, 0)
    y_pred_reshaped = y_pred_reshaped.replace(negative_label, 0)
    y_true_reshaped = y_true_reshaped.replace(positive_labels, 1)
    y_pred_reshaped = y_pred_reshaped.replace(positive_labels, 1)

    # Split the positive predictions and the negative predictions
    positive_predictions = y_true_reshaped[y_true_reshaped != 0] - y_pred_reshaped[y_true_reshaped != 0]
    negative_predictions = y_true_reshaped[y_true_reshaped == 0] - y_pred_reshaped[y_true_reshaped == 0]

    # tp/tn/fp/fn
    TP = len(positive_predictions[positive_predictions == 0])
    TN = len(negative_predictions[negative_predictions == 0])
    FP = len(negative_predictions[negative_predictions != 0])
    FN = len(positive_predictions[positive_predictions != 0])

    # recall and precision
    outputs = {
        'recall': float(TP) / (TP + FN) if (TP + FN) != 0 else 1.0,
        'precision': float(TP) / (TP + FP) if (TP + FP) != 0 else 1.0
    }

    # F beta score
    if outputs['precision'] == 0 and outputs['recall'] == 0:
        outputs['F_beta'] = 0
    else:
        outputs['F_beta'] = (1 + np.square(beta)) * outputs['precision'] * outputs['recall'] / \
                            (np.square(beta) * outputs['precision'] + outputs['recall'])
    if isinstance(return_value, str):
        if return_value not in outputs:
            raise Exception("Unknown return value: " + return_value)
        return outputs[return_value]

    elif isinstance(return_value, list):
        return_output = {}
        for value in return_value:
            if value not in outputs:
                raise Exception("Unknown value in 'return_value' list!")
            else:
                return_output[value] = outputs[value]
        return return_output

    else:
        raise Exception("Unknown type for 'return_value'!")