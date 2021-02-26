import logging
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from pandas.core.dtypes.common import ensure_float64
from sklearn.metrics import precision_score, recall_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


def _create_evaluate_df_results(evaluations_binaries):
    results = pd.DataFrame()
    results['feature'] = evaluations_binaries.keys()
    results['feature_original_name'] = [evaluations_binaries[feature].get('feature_original_name')
                                        for feature in evaluations_binaries]
    results['baseline_precision'] = [evaluations_binaries[feature].get('baseline_precision')
                                     for feature in evaluations_binaries]
    results['count_1'] = [evaluations_binaries[feature].get('count_1_in_x')
                          for feature in evaluations_binaries]
    results['precision_by_1'] = [evaluations_binaries[feature].get('precision_by_1')
                                 for feature in evaluations_binaries]
    results['recall_by_1'] = [evaluations_binaries[feature].get('recall_by_1')
                              for feature in evaluations_binaries]
    results['count_0'] = [evaluations_binaries[feature].get('count_0_in_x')
                          for feature in evaluations_binaries]
    results['precision_by_0'] = [evaluations_binaries[feature].get('precision_by_0')
                                 for feature in evaluations_binaries]
    results['recall_by_0'] = [evaluations_binaries[feature].get('recall_by_0')
                              for feature in evaluations_binaries]
    return results


def evaluate_df_with_binary_output(X_df, y, pos_label=1, categorical_features=None, bins_dict=None):
    if not isinstance(X_df, pd.DataFrame):
        raise Exception('df argument should be pandas.DataFrame only')

    if not categorical_features:
        categorical_features = []
    if not bins_dict:
        bins_dict = {}

    if isinstance(y, pd.DataFrame):
        y = np.asarray(y[y.columns[0]])
    else:
        y = np.asarray(y)

    results = {}
    y_column = ensure_float64(y)
    i = 0
    for col in X_df.columns:
        is_categorical = True if col in categorical_features else False
        bins = []
        if col in bins_dict.keys():
            bins = bins_dict[col]
        c = _evaluate_binary_output(
            X_df[[col]], y_column, pos_label=pos_label, is_categorical=is_categorical, bins=bins)
        results.update(c)
        i += 1
    results_df = _create_evaluate_df_results(results)
    return results_df


def _evaluate_binary_output(x, y, pos_label=1, is_categorical=False, bins=None):
    try:
        if len(x.columns) > 1:
            raise Exception("need only one column in x")

        if not bins:
            bins = []

        result = {}
        feature_original_name = x.columns[0]

        if is_categorical:
            x[x.columns[0]] = x[x.columns[0]].astype(object)
            dummies = pd.get_dummies(x, prefix=x.columns[0], prefix_sep='_|_')
            for d in dummies.columns:
                result[d] = _evaluate_binary_feature(dummies[d], y, feature_original_name, pos_label)
            return result

        if len(x[x.columns[0]].unique()) == 1:
            for col in x.columns:
                result[col] = {}
            return result

        if len(x[x.columns[0]].unique()) == 2:
            x_max = max(x[x.columns[0]])
            x[x.columns[0]] = x[x.columns[0]].apply(lambda n: 1 if x_max == n else 0)

            for col in x.columns:
                result[col] = _evaluate_binary_feature(x, y, feature_original_name, pos_label)
            return result
        # continuous variable
        else:
            if len(bins) == 0:
                result[x.columns[0]] = {}
            else:
                cur_bins = bins
                labels = range(len(cur_bins) - 1)
                print(cur_bins)
                categorical_x = pd.DataFrame(data=pd.cut(x[x.columns[0]].apply(float), bins=cur_bins,
                                             labels=labels), columns=x.columns)
                result = _evaluate_binary_output(categorical_x, y, pos_label, is_categorical=True)
            return result
    except Exception as e:
        raise e


def _evaluate_binary_feature(x, y, feature_original_name, pos_label=1):
    x = ensure_float64(x)
    y = ensure_float64(y)
    if not len(np.unique(x)) <= 2 or not len(np.unique(y)):
        raise Exception("please provide binary x and y")
    y_positive_rate = np.count_nonzero(y == pos_label)/float(len(y))
    count_0 = np.count_nonzero(x == 0)
    count_1 = np.count_nonzero(x == 1)
    precision_0 = precision_score(y, 1.0 - x, pos_label=pos_label)
    recall_0 = recall_score(y, 1.0 - x, pos_label=pos_label)
    precision_1 = precision_score(y, x, pos_label=pos_label)
    recall_1 = recall_score(y, x, pos_label=pos_label)
    is_0_informative = precision_0 > y_positive_rate
    is_1_informative = precision_1 > y_positive_rate

    metrics = {'feature_original_name': feature_original_name,
               'baseline_precision': y_positive_rate,
               'count_0_in_x': count_0,
               'count_1_in_x': count_1,
               'precision_by_0' : precision_0,
               'recall_by_0' : recall_0,
               'precision_by_1' : precision_1,
               'recall_by_1' : recall_1,
               'is_0_informative': is_0_informative,
               'is_1_informative': is_1_informative}
    return metrics


def polynomial_features_labeled(raw_input_df, power):
    """
        This is a wrapper function for sklearn's Ploynomial features, which returns the resulting power-matrix with
        meaningful labels. It calls sklearn.preprocessing.PolynomialFeatures on raw_input_df with the specified power
        parameter and return it's resulting array as a labeled pandas dataframe (i.e cols= ['a', 'b', 'a^2', 'axb', 'b^2']).
        :param raw_input_df: labeled pandas dataframe.
        :param power: The degree of the polynomial features (use the same power as you want entered into
        PolynomialFeatures(power) directly).
        :return: power-matrix with meaningful column labels.
    """
    input_df = raw_input_df.copy()
    poly = PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s^%d" % (variable, power)
                if final_label == "":
                    final_label = intermediary_label
                else:
                    final_label = final_label + " x " + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns=target_feature_names, index=input_df.index)
    return output_df


def lag_feature(feature_df, lag, dropna=False):
    """
        Shifts feature forward by a specified lag. first n=lag values will be None.
        :param feature_df: A pandas series.
        :param lag: int. lag size.
        :param dropna: performs dropna after shift. default-False.
        :return: Transformed feature with the lag value added to the column name (i.e col_name_lag_5)..
    """
    res = pd.DataFrame(columns=['%s_lag_%s' % (feature_df.name, str(lag))],
                       data=feature_df.shift(lag, fill_value=None).values,
                       index=feature_df.index)
    if dropna:
        return res.dropna()
    else:
        return res


def lead_feature(feature_df, lead, dropna=False):
    """
      Shifts feature backward by a specified lead. last n=lead values will be None.
      :param feature_df: A pandas series.
      :param lead: int. lead size.
      :param dropna: performs dropna after shift. default-False.
      :return: Transformed feature with the lead value added to the column name (i.e col_name_lead_5).
    """
    res = pd.DataFrame(columns=['%s_lead_%s' % (feature_df.name, str(lead))],
                       data=feature_df.shift(-lead, fill_value=None).values,
                       index=feature_df.index)
    if dropna:
        return res.dropna()
    else:
        return res


def log_feature(feature_df):
    """
         Returns a natural log transformation of the feature.
         :param feature_df: A pandas series.
         :return: Transformed feature with 'log' added to the column name (i.e col_name_log).
    """
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_log' % feature_df.name, inplace=True)
        return feature_df_bis.apply(np.log)
    except:
        logger.info('Not possible to log', feature_df.name)
        return feature_df


def exponent_feature(feature_df):
    """
         Returns a natural exponent transformation of the feature.
         :param feature_df: A pandas series.
         :return: Transformed feature with 'exp' added to the column name (i.e col_name_exp).
    """
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_exp' % feature_df.name, inplace=True)
        return feature_df_bis.apply(np.exp)
    except:
        logger.info('Not possible to exponent', feature_df.name)
        return feature_df


def power_feature(feature_df, power):
    """
         Returns the feature raised to the power specified.
         :param feature_df: A pandas series.
         :param power: int.
         :return: Transformed feature with power value added to the column name (i.e col_name_pow_5).
    """
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_pow_%s' % (feature_df.name, str(power)), inplace=True)
        return feature_df_bis.apply(lambda x: np.power(x, power))
    except:
        logger.info('Not possible to power {0}'.format(power), feature_df.name)
        return feature_df


def sqrt_feature(feature_df):
    """
        Returns a square root transformation of the feature.
        :param feature_df: A pandas series.
        :return: Transformed feature with 'sqrt' added to the column name (i.e col_name_sqrt).
   """
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_sqrt' % feature_df.name, inplace=True)
        return feature_df_bis.apply(np.sqrt)
    except:
        logger.info('Not possible to sqrt', feature_df.name)
        return feature_df


def inverse_feature(feature_df):
    """
        Returns the inverse (1/x) transformation of the feature.
        :param feature_df: A pandas series.
        :return: Transformed feature with 'inverse' added to the column name (i.e col_name_inverse).
   """
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_inverse' % feature_df.name, inplace=True)
        return feature_df_bis.apply(lambda x: float(1) / x)
    except:
        logger.info('Not possible to inverse', feature_df.name)
        return feature_df


def interact_features(feature_df, interact_list, drop_original_columns=True):
    """
    This function create interactions between pairs of features
    :param feature_df: a pandas dataframe
    :param interact_list: list of lists or tuples with two strings each, representing columns to be interacted.
    :param drop_original_columns: (bool) if set to True, columns to be interacted will be droped from the dataframe. note-
    if set to true a column cannot appear in more then one interaction pair.
    :return: DataFrame with the interactions columns, without the original columns.
    """

    for features_pair in interact_list:
        feature_0 = features_pair[0]
        feature_1 = features_pair[1]

        if (feature_0 not in feature_df.columns) or (feature_1 not in feature_df.columns):
            print('Warning: one the features: ' + feature_0 + ',' + feature_1 + 'do not exists in the Data.')
        else:
            new_feature = feature_df[feature_0] * feature_df[feature_1]
            if drop_original_columns:
                feature_df = feature_df.drop(feature_0, axis=1)
                feature_df = feature_df.drop(feature_1, axis=1)
            feature_df[feature_0 + '_X_' + feature_1] = new_feature

    return feature_df


def _remove_duplicate_columns(df):
    """
    find duplicate column names and keep only the first column.
    :param df: pandas dataframe.
    :return: pandas dataframe with duplicates removed.
    """
    cols = list(df.columns)
    to_drop = False
    for i, item in enumerate(df.columns):
        if item in df.columns[:i]:
            cols[i] = "toDROP"
            to_drop = True
    df.columns = cols
    if to_drop:
        df = df.drop("toDROP", 1)
    return df


def extend_dataframe(df, y_name_col=None, index_col=None, lead_order=3, lag_order=3, power_order=2, log=True, exp=True, sqrt=True, poly_degree=2, dropna =False, fillna_value=0, inverse=True):
    """
    Create a new dataframe with transformed features added as new columns.
    :param df: initial dataframe
    :param y_name_col: y column name. this column will not be extended.
    :param index_col: index column/s (string or list of strings). Any column that should NOT be extended, can be listed here.
    :param lead_order: shifts the feature values back. Extended dataframe will include all leads from 1 to lead_order specified.
    :param lag_order: shifts the feature values forward. Extended dataframe will include all lags from 1 to lead_order specified.
    :param power_order: Extended dataframe will include all powers from 1 to power_order specified.
    :param log: perform natural log transformation. Default=True.
    :param exp: perform natural exponent transformation. Default=True.
    :param sqrt: perform square root tranformation. Default=True.
    :param poly_degree: the highest order polynomial for the transformation. 0=no polynomial transformation.
    :param dropna: default True.
    :param fillna_value: default True.
    :param inverse: perform inverse transformation (1/x). default=True.
    :return: extended dataframe.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    not_to_extend = []
    if index_col is not None:
        if type(index_col) is list:
            not_to_extend += [ix_col for ix_col in index_col if ix_col in df.columns]
        elif index_col in df.columns:
            not_to_extend += [index_col]
    if y_name_col is not None and y_name_col in df.columns:
        not_to_extend += [y_name_col]

    features_columns = df.columns[~df.columns.isin(not_to_extend)]
    result = df[features_columns].copy()

    if isinstance(df, pd.DataFrame):
        for feature_name in features_columns:
            feature_df = df[feature_name]
            if lead_order > 0:
                for lead in range(1, lead_order + 1):
                    transformed_feature = lead_feature(feature_df, lead)
                    transformed_feature.rename(columns={feature_name: 'lead_' + feature_name}, inplace=True)
                    result = pd.concat([result, transformed_feature], axis=1)
            if lag_order > 0:
                for lag in range(1, min(lag_order, len(df)) + 1):
                    transformed_feature = lag_feature(feature_df, lag)
                    transformed_feature.rename(columns={feature_name: 'lag_' + feature_name}, inplace=True)
                    result = pd.concat([result, transformed_feature], axis=1)
            if power_order > 1:
                for power in range(2, power_order + 1):
                    transformed_feature = power_feature(feature_df, power)
                    result['power_' + feature_name] = transformed_feature
                    # transformed_feature.rename(columns={feature_name: 'power_' + feature_name}, inplace=True)
                    # result = pd.concat([result, transformed_feature], axis=1)
            if log > 0:
                if (feature_df >= 1).all():
                    transformed_feature = log_feature(feature_df)
                    result['log_' + feature_name] = transformed_feature
                    result.replace(-np.inf, 0, inplace=True)
            if exp > 0:
                if (feature_df < 100).all():
                    transformed_feature = exponent_feature(feature_df)
                    result['exp_' + feature_name] = transformed_feature
            if sqrt > 0:
                if (feature_df > 0).all():
                    transformed_feature = sqrt_feature(feature_df)
                    result['sqrt_' + feature_name] = transformed_feature
                    result.replace(-np.inf, 0, inplace=True)
            if inverse:
                if (feature_df != 0).all():
                    transformed_feature = inverse_feature(feature_df)
                    result['inverse_' + feature_name] = transformed_feature

        if poly_degree > 0:
            transformed_feature = polynomial_features_labeled(df[features_columns], poly_degree)
            result = pd.concat([result, transformed_feature], axis=1)
        final_extended_df = _remove_duplicate_columns(result)

        if dropna:
            final_extended_df.dropna(inplace=True)
        else:
            final_extended_df.fillna(fillna_value, inplace=True)

        if len(not_to_extend) > 0:
            final_extended_df = pd.concat([df[not_to_extend], final_extended_df], axis=1)

        return final_extended_df

    else:
        logger.error('Only works for DataFrame or Series')
        return None


def _calc_se(X, y, y_hat, DOF):
    """
    calc the formula: SE = sqrt(C.diagonal)
    C = MSE * (X'X)^-1 is the variance-covariance matrix
    MSE = SSE / DOF, Where  SSE = sum((y - y_hat)^2) and DF = n - k - 1 where
    n = number of obsevations, k = number of coefficients
    X = features matrix. X' = X transpose matrix
    @param X: Dataframe or Series
    @param y: Dataframe or Series
    @param y_hat: Dataframe or Series
    @param DOF: int - degree of freedom
    @return: numpay array
    """
    if type(y) is pd.DataFrame: y = y[y.columns[0]]
    SSE = np.sum(np.power(y - y_hat, 2))
    MSE = np.true_divide(SSE, DOF)
    X_transpose = pd.DataFrame.transpose(X)
    X_mul_X_transpose = np.dot(X_transpose.values, X.values)
    C = np.linalg.inv(X_mul_X_transpose)
    X_diagonal = np.diagonal(C)
    SE = pd.Series(np.sqrt(MSE * X_diagonal))
    return SE


def calc_t_values(X, y, y_hat, coefficients):
    """
    p-values for regerssion assets
    calc t values: t(b_i) = b_i / SE(b_i)
    Where b_i is the beta (coefficient) of x_i
    and SE(b_i) is the standard error of the coefficient
    @param X: Dataframe or Series
    @param y: Dataframe or Series
    @param y_hat: Dataframe or Series
    @param coefficients: list or ndarray
    @return: numpay array
    """
    if X.shape[1] != len(coefficients):
        raise Exception('X shape ' + repr(X.shape) + ' not match coefficients length ' + repr(len(coefficients)))

    # degree of freedom
    DOF = len(X) - len(coefficients) - 1
    if DOF < 1:
        raise Exception('Degrees Of Freedom must be greater or equals to 1')

    SE = _calc_se(X, y, y_hat, DOF)
    SE = pd.Series([1 if e == 0 else e for e in SE])
    t_statistics = np.true_divide(coefficients, SE)
    return t_statistics


def calc_p_values(X, y, y_hat, coefficients):
    """
    p-values for regression assets. Calc p values from t values
    @param X: pd.DataFrame or pd.Series
    @param y: pd.DataFrame or pd.Series
    @param y_hat: pd.DataFrame or pd.Series
    @param coefficients: list or ndarray
    @return: NumPy array
    """
    if X.shape[1] != len(coefficients):
        raise Exception('X shape ' + repr(X.shape) + ' not match coefficients length ' + repr(len(coefficients)))

    # degree of freedom
    DOF = len(X) - len(coefficients) - 1
    if DOF < 1:
        raise Exception('Degrees Of Freedom must be greater or equals to 1')

    t_statistics = calc_t_values(X, y, y_hat, coefficients)
    p_values = map(lambda t: 2 * stats.t.sf(t, DOF), np.abs(t_statistics))
    return list(p_values)
