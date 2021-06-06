import logging
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from pandas.core.dtypes.common import ensure_float64
from sklearn.metrics import precision_score, recall_score, f1_score
import traceback
from scipy.sparse import csc_matrix, lil_matrix, diags
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

def _create_results(evaluations_binaries, store_results,output_path):
    results = pd.DataFrame()
    results['feature'] = evaluations_binaries.keys()
    results['feature_original_name'] = [evaluations_binaries[feature].get('feature_original_name') for feature in evaluations_binaries]
    results['baseline_precision'] = [evaluations_binaries[feature].get('baseline_precision') for feature in evaluations_binaries]
    results['count_1'] = [evaluations_binaries[feature].get('count_1_in_x') for feature in evaluations_binaries]
    results['precision_by_1'] = [evaluations_binaries[feature].get('precision_by_1') for feature in evaluations_binaries]
    results['recall_by_1'] = [evaluations_binaries[feature].get('recall_by_1') for feature in evaluations_binaries]
    results['count_0'] = [evaluations_binaries[feature].get('count_0_in_x') for feature in evaluations_binaries]
    results['precision_by_0'] = [evaluations_binaries[feature].get('precision_by_0') for feature in evaluations_binaries]
    results['recall_by_0'] = [evaluations_binaries[feature].get('recall_by_0') for feature in evaluations_binaries]
    if output_path is not None and isinstance(output_path,str) and store_results:
        results.to_csv(output_path, index=False)
    return results

def evaluate_df_with_binary_output(X_df, y, pos_label=1, categorical_features=[], bins_dict = {}, store_results=True, output_path=None):

    if not isinstance(X_df, pd.DataFrame):
        raise Exception('df argument should be pandas.DataFrame only')
    # mat = X_df.values
    # mat = ensure_float64(mat).T
    # mask = np.isfinite(mat)
    # K = len(X_df.columns)

    if isinstance(y, pd.DataFrame):
        y = np.asarray(y[y.columns[0]])
    else:
        y = np.asarray(y)

    results = {}
    y_column = ensure_float64(y)
    mask_y = np.isfinite(y_column)
    i=0
    try:
        for col in X_df.columns:
            is_categorical = True if col in categorical_features else False
            bins = []
            if col in bins_dict.keys():
                bins = bins_dict[col]
            # valid = mask[i] & mask_y
            # if valid.sum() < 1:
            #     c = {col:{}}
            # elif not valid.all():
            #     c = evaluate_binary_output(X_df[col][valid],y_column[valid], pos_label= pos_label,is_categorical=is_categorical, bins=bins)
            # else:
            c = evaluate_binary_output(X_df[[col]], y_column, pos_label= pos_label, is_categorical=is_categorical,  bins=bins)
            results.update(c)
            i+=1
        results_df = _create_results(results, store_results, output_path)
        return results_df
    except Exception as e:
        traceback.print_exc()

def evaluate_binary_output(x,y, pos_label=1, is_categorical=False, bins=[]):
    try:
        if len(x.columns)>1:
            raise Exception("need only one column in x")
        result = {}
        feature_original_name = x.columns[0]
        if is_categorical:
            x[x.columns[0]] = x[x.columns[0]].astype(object)
            dummies = pd.get_dummies(x, prefix=x.columns[0])
            for d in dummies.columns:
                result[d] = evaluate_binary_feature(dummies[d],y,feature_original_name,pos_label)
            return result
        if len(x[x.columns[0]].unique())==1:
            for col in x.columns:
                result[col] = {}
            return result
        if len(x[x.columns[0]].unique())==2:
            x_max = max(x[x.columns[0]])
            x[x.columns[0]] = x[x.columns[0]].apply(lambda n: 1 if x_max == n else 0)
            values = np.unique(x)
            #x = [0 if i==values[0] else 1 for i in x]
            for col in x.columns:
                result[col] = evaluate_binary_feature(x,y,feature_original_name,pos_label)
            return result
        else: #continous variable

            if len(bins)==0:
                # print(str(x.columns[0]) + " is continuos but you did not provide bins, using random 3 bins!")
                # cur_bins = 5
                # labels = range(5)
                for col in x.columns:
                    result[col] = {}
                    return result
            else:
                cur_bins = bins
                labels = range(len(cur_bins) - 1)
            print(cur_bins)
            categorical_x = pd.DataFrame(data=pd.cut(x[x.columns[0]].apply(float), bins=cur_bins, labels=labels), columns=x.columns)
            results = evaluate_binary_output(categorical_x,y, pos_label, is_categorical=True)
            return results


    except Exception as e:
        raise e

def evaluate_binary_feature(x,y, feature_original_name,pos_label=1):
    x = ensure_float64(x)
    y = ensure_float64(y)
    if not len(np.unique(x)) <= 2 or not len(np.unique(y)):
        raise Exception("please provide binary x and y")
    y_positive_rate = np.count_nonzero(y==pos_label)/float(len(y))

    count_0=np.count_nonzero(x==0)
    count_1= np.count_nonzero(x==1)
    precision_0 =precision_score(y,1.0-x,pos_label=pos_label)
    recall_0 = recall_score(y,1.0-x,pos_label=pos_label)
    precision_1 = precision_score(y,x,pos_label=pos_label)
    recall_1 = recall_score(y,x,pos_label=pos_label)
    is_0_informative = precision_0>y_positive_rate
    is_1_informative = precision_1>y_positive_rate

    metrics = {
        'feature_original_name':feature_original_name,
        'baseline_precision': y_positive_rate,
        'count_0_in_x': count_0,
        'count_1_in_x': count_1,
        'precision_by_0' : precision_0,
        'recall_by_0' : recall_0,
        'precision_by_1' : precision_1,
        'recall_by_1' : recall_1,
        'is_0_informative':is_0_informative,
        'is_1_informative': is_1_informative

    }
    return metrics

def bi_corr(x1, x2):
    '''
    :param x1: first feature to correlate
    :param x2: second feature to correlate
    :return: correlation of x1,x2. if both are binary - we calculate spearmanr(x1,x2),
    if both are continous - we calculate pearsonr(x1,x2), if one is binary and one is continous
    we calculate the pointbiserial correlation in the right order
    '''
    x1 = ensure_float64(x1)
    x2 = ensure_float64(x2)
    if len(np.unique(x1)) <= 2:
        if len(np.unique(x2)) <= 2:
            return stats.spearmanr(x1, x2, nan_policy='omit')
        else:
            return stats.pointbiserialr(x1, x2)
    else:
        if len(np.unique(x2)) <= 2:
            return stats.pointbiserialr(x2, x1)
        else:
            return stats.pearsonr(x1, x2)


def correlate_features(df, y=None):
    '''
    :param df: the dataframe containing features to correlate
    :param y: if not None, the function returns the correlation of any feature in df with y
    :return: correlation matrix in dataFrame, or correlation dictionary of every column with Y
    '''
    if not isinstance(df, pd.DataFrame):
        raise Exception('df argument should be pandas.DataFrame only')
    mat = df.values
    mat = ensure_float64(mat).T
    mask = np.isfinite(mat)
    K = len(df.columns)

    if y is None:
        correlation_matrix = np.empty([K,K], dtype=float)
        for i, col_a in enumerate(mat):
            for j, col_b in enumerate(mat):
                if i > j: # we are in the lower triangle
                    continue
                valid = mask[i] & mask[j]
                if valid.sum() < 1:
                    c = [np.nan]
                elif i == j:
                    c = [1.]
                elif not valid.all():
                    c = bi_corr(col_a[valid], col_b[valid])
                else:
                    c = bi_corr(col_a, col_b)
                correlation_matrix[i, j] = c[0]
                correlation_matrix[j, i] = c[0]
        corr_df = pd.DataFrame(data=correlation_matrix, index=df.columns, columns=df.columns)
        return corr_df
    else:   # we only want correlation with y column
        if isinstance(y, pd.DataFrame):
            y = np.asarray(y[y.columns[0]])
        else:
            y = np.asarray(y)
        correlation_array = {}
        y_column = ensure_float64(y)
        mask_y = np.isfinite(y_column)
        for i, col_a in enumerate(mat):
                valid = mask[i] & mask_y
                if valid.sum() < 1:
                    c = np.nan
                elif not valid.all():
                    c = bi_corr(col_a[valid],y_column[valid])
                else:
                    c = bi_corr(col_a, y_column)
                correlation_array[df.columns[i]] = c[0]
        return correlation_array


def polynomial_features_labeled(raw_input_df, power):
    '''
    :param input_df
    input_df = Your labeled pandas dataframe (list of x's not raised to any power)

    :param power
    power = what order polynomial you want variables up to. (use the same power as you want entered into
    pp.PolynomialFeatures(power) directly)

    :return output_df:
    Output: This function relies on the powers_ matrix which is one of the preprocessing function's outputs to create
    logical labels and outputs a labeled pandas dataframe
    '''
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


def lag_feature(feature_df, periods_of_lag, dropna=False):
    values = feature_df.values
    lag_values = [values[val_index] for val_index in range(len(values) - periods_of_lag)]
    arranged_lag_values = [None] * periods_of_lag + lag_values

    res = pd.DataFrame(index=feature_df.index,
                       columns=['%s_lag_%s' % (feature_df.name, str(periods_of_lag))],
                       data=arranged_lag_values)

    if dropna:
        return res.dropna()
    else:
        return res


def lead_feature(feature_df, lead, dropna=False):
    res = pd.DataFrame(columns=['%s_lead_%s' % (feature_df.name, str(lead))],
                       data=feature_df.shift(-lead).values,
                       index=feature_df.index)
    if dropna:
        return res.dropna()
    else:
        return res


def log_feature(feature_df):
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_log' % feature_df.name, inplace=True)
        return feature_df_bis.apply(np.log)
    except:
        logger.info('Not possible to log', feature_df.name)
        return feature_df


def exponent_feature(feature_df):
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_exp' % feature_df.name, inplace=True)
        return feature_df_bis.apply(np.exp)
    except:
        logger.info('Not possible to expoenent', feature_df.name)
        return feature_df


def power_feature(feature_df, power):
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_pow_%s' % (feature_df.name, str(power)), inplace=True)
        return feature_df_bis.apply(lambda x: np.power(x, power))
    except:
        logger.info('Not possible to power {0}'.format(power), feature_df.name)
        return feature_df


def sqrt_feature(feature_df):
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_sqrt' % feature_df.name, inplace=True)
        return feature_df_bis.apply(np.sqrt)
    except:
        logger.info('Not possible to sqrt', feature_df.name)
        return feature_df


def inverse_feature(feature_df):
    feature_df_bis = feature_df.copy()
    try:
        feature_df_bis.rename('%s_reverse' % feature_df.name, inplace=True)
        return feature_df_bis.apply(lambda x: float(1) / x)
    except:
        logger.info('Not possible to revert', feature_df.name)
        return feature_df


def remove_duplicate_columns(df):
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


def extend_dataframe(df, y_name_col=None, index_col=[], lead_order=3, lag_order=3, power_order=2, log=1, exp=1, sqrt=1, poly_degree=2, dropna =False, fillna_value=0, inverse=True):
    '''
    Create a new dataframe with transformed features: lead, log, lag, exp, pow. Each transformation should be initiated
    to 0 if you don't want this transformation to appear

    :param df: initial dataframe
    :param y_name_col: column not to extend
    :param date_col: if date column exists and we do not want to extend it
    :param lead_order: Would include all lead from 1 to lead_order
    :param lag_order: Would include all lag from 1 to lag_order
    :param power_order: Would include all power from 1 to power_order
    :param log: 0 (not included) or 1(include)
    :param exp: 0 (not included) or 1(include)
    :param sqrt: 0 (not included) or 1(include)
    :return: extended dataframe
    '''
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # if y_name_col:
    #     features_columns = [column for column in df.columns if column != y_name_col]
    # else:
    if type(index_col) is list :
        not_to_extend = index_col + [y_name_col]
    else:
        not_to_extend = [index_col, y_name_col]

    features_columns = df.columns[~df.columns.isin(not_to_extend)]

    if poly_degree > 0:
        result = pd.DataFrame()
    else:
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
                # transformed_feature.rename(columns={feature_name: 'log_' + feature_name}, inplace=True)
                # result = pd.concat([result, transformed_feature], axis=1)
            if exp > 0:
                if (feature_df < 100).all():
                    transformed_feature = exponent_feature(feature_df)
                    result['exp_' + feature_name] = transformed_feature
                # transformed_feature.rename(columns={feature_name: 'exp_' + feature_name}, inplace=True)
                # result = pd.concat([result, transformed_feature], axis=1)
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
        # print len(result.columns)
        final_extended_df = remove_duplicate_columns(result)
        # print len(final_extended_df.columns)

        if dropna:
            final_extended_df.dropna(inplace=True)
        else:
            final_extended_df.fillna(fillna_value, inplace=True)

        if y_name_col and y_name_col in df.columns:
            final_extended_df = pd.concat([result, df[y_name_col]], axis=1)

        if index_col and (ix_col in df.columns for ix_col in index_col):
            final_extended_df = pd.concat([result, df[index_col]], axis=1)
        return final_extended_df

    else:
        logger.error('Only works for DataFrame or Series')
        return None


def standardize_features(X):
    '''
    Builds a new DataFrame to facilitate regressing over standardized features
    '''
    scaler_model = sklearn.preprocessing.StandardScaler()
    X_norm = scaler_model.fit_transform(X)
    return scaler_model, pd.DataFrame(X_norm, index=X.index)


def interpolate_quaterly_to_monthly(quaterly_data, show_img=False):
    time_range = [i * float(1) / 3 for i in range(1, len(quaterly_data) + 1)]
    f2 = interp1d(time_range, quaterly_data, kind='cubic')
    if show_img:
        plt.plot(range(len(quaterly_data)), quaterly_data, 'o')
        plt.show()
    return f2([i * 1 / 12 for i in range(4, int(12 * time_range[len(time_range) - 1]))])


def interpolate_monthly_to_daily(df, interpolate_method='linear', order=2):
    '''

    :param df: dataframe with one column for interpolation and datetime index on monthly basis
    :param inertpolate_method: 'linear', or 'spline'
    :param order: if 'spline' method is chosen, order should be defined. by default its cubic spline
    :return: interpolated dataframe array
    '''
    if interpolate_method == 'linear':
        df2 = df.resample('D').interpolate(method='linear')
    if interpolate_method == 'spline':
        df2 = df.resample('D').interpolate(method='spline', order=order)
    return df2


#TODO: This seems to not work. Check with Keren
def __calc_se(X, y, y_hat, DOF):
    # calc the formula: SE = sqrt(C.diagonal)
    # C = MSE * (X'X)^-1 is the variance-covariance matrix
    # MSE = SSE / DOF, Where  SSE = sum((y - y_hat)^2) and DF = n - k - 1 where
    # n = number of obsevations, k = number of coefficients
    # X = features matrix. X' = X transpose matrix
    if type(y) is pd.DataFrame: y = y[y.columns[0]]
    SSE = np.sum(np.power(y - y_hat, 2))
    MSE = np.true_divide(SSE, DOF)
    X_transpose = pd.DataFrame.transpose(X)
    X_mul_X_transpose = np.dot(X_transpose.values, X.values)
    C = np.linalg.inv(X_mul_X_transpose)
    X_diagonal = np.diagonal(C)
    SE = pd.Series(np.sqrt(MSE * X_diagonal))
    return SE


#TODO: This seems to not work. Check with Keren
def calc_t_values(X, y, y_hat, coefficients):  # p-values for regerssion models
    # calc t values: t(b_i) = b_i / SE(b_i)
    # Where b_i is the beta (coefficient) of x_i
    # and SE(b_i) is the standard error of the coefficient
    if X.shape[1] != len(coefficients):
        raise Exception('X shape ' + repr(X.shape) + ' not match coefficients length ' + repr(len(coefficients)))

    # degree of freedom
    DOF = len(X) - len(coefficients) - 1
    if DOF < 1:
        raise Exception('Degrees Of Freedom must be greater or equals to 1')

    SE = __calc_se(X, y, y_hat, DOF)
    SE = pd.Series([1 if e == 0 else e for e in SE])
    t_statistics = np.true_divide(coefficients, SE)
    return t_statistics


#TODO: This seems to not work. Check with Keren
def calc_p_values(X, y, y_hat, coefficients):
    '''
    p-values for regression models. Calc p values from t values
    '''
    if X.shape[1] != len(coefficients):
        raise Exception('X shape ' + repr(X.shape) + ' not match coefficients length ' + repr(len(coefficients)))

    # degree of freedom
    DOF = len(X) - len(coefficients) - 1
    if DOF < 1:
        raise Exception('Degrees Of Freedom must be greater or equals to 1')

    t_statistics = calc_t_values(X, y, y_hat, coefficients)
    p_values = map(lambda t: 2 * stats.t.sf(t, DOF), np.abs(t_statistics))
    return p_values

def construct_W(X, **kwargs):
    """
    Construct the affinity matrix W through different ways
    Notes
    -----
    if kwargs is null, use the default parameter settings;
    if kwargs is not null, construct the affinity matrix according to parameters in kwargs
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        parameters to construct different affinity matrix W:
        y: {numpy array}, shape (n_samples, 1)
            the true label information needed under the 'supervised' neighbor mode
        metric: {string}
            choices for different distance measures
            'euclidean' - use euclidean distance
            'cosine' - use cosine distance (default)
        neighbor_mode: {string}
            indicates how to construct the graph
            'knn' - put an edge between two nodes if and only if they are among the
                    k nearest neighbors of each other (default)
            'supervised' - put an edge between two nodes if they belong to same class
                    and they are among the k nearest neighbors of each other
        weight_mode: {string}
            indicates how to assign weights for each edge in the graph
            'binary' - 0-1 weighting, every edge receives weight of 1 (default)
            'heat_kernel' - if nodes i and j are connected, put weight W_ij = exp(-norm(x_i - x_j)/2t^2)
                            this weight mode can only be used under 'euclidean' metric and you are required
                            to provide the parameter t
            'cosine' - if nodes i and j are connected, put weight cosine(x_i,x_j).
                        this weight mode can only be used under 'cosine' metric
        k: {int}
            choices for the number of neighbors (default k = 5)
        t: {float}
            parameter for the 'heat_kernel' weight_mode
        fisher_score: {boolean}
            indicates whether to build the affinity matrix in a fisher score way, in which W_ij = 1/n_l if yi = yj = l;
            otherwise W_ij = 0 (default fisher_score = false)
        reliefF: {boolean}
            indicates whether to build the affinity matrix in a reliefF way, NH(x) and NM(x,y) denotes a set of
            k nearest points to x with the same class as x, and a different class (the class y), respectively.
            W_ij = 1 if i = j; W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y) (default reliefF = false)
    Output
    ------
    W: {sparse matrix}, shape (n_samples, n_samples)
        output affinity matrix W
    """

    # default metric is 'cosine'
    if 'metric' not in kwargs.keys():
        kwargs['metric'] = 'cosine'

    # default neighbor mode is 'knn' and default neighbor size is 5
    if 'neighbor_mode' not in kwargs.keys():
        kwargs['neighbor_mode'] = 'knn'
    if kwargs['neighbor_mode'] == 'knn' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'k' not in kwargs.keys():
        kwargs['k'] = 5
    if kwargs['neighbor_mode'] == 'supervised' and 'y' not in kwargs.keys():
        print ('Warning: label is required in the supervised neighborMode!!!')
        exit(0)

    # default weight mode is 'binary', default t in heat kernel mode is 1
    if 'weight_mode' not in kwargs.keys():
        kwargs['weight_mode'] = 'binary'
    if kwargs['weight_mode'] == 'heat_kernel':
        if kwargs['metric'] != 'euclidean':
            kwargs['metric'] = 'euclidean'
        if 't' not in kwargs.keys():
            kwargs['t'] = 1
    elif kwargs['weight_mode'] == 'cosine':
        if kwargs['metric'] != 'cosine':
            kwargs['metric'] = 'cosine'

    # default fisher_score and reliefF mode are 'false'
    if 'fisher_score' not in kwargs.keys():
        kwargs['fisher_score'] = False
    if 'reliefF' not in kwargs.keys():
        kwargs['reliefF'] = False

    n_samples, n_features = np.shape(X)

    # choose 'knn' neighbor mode
    if kwargs['neighbor_mode'] == 'knn':
        k = kwargs['k']
        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                # compute pairwise euclidean distances
                D = pairwise_distances(X)
                D **= 2
                # sort the distance matrix D in ascending order
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                # choose the k-nearest neighbors for each instance
                idx_new = idx[:, 0:k+1]
                G = np.zeros((n_samples*(k+1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order='F')
                G[:, 2] = 1
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

            elif kwargs['metric'] == 'cosine':
                # normalize the data first
                X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
                # compute pairwise cosine distances
                D_cosine = np.dot(X, np.transpose(X))
                # sort the distance matrix D in descending order
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0:k+1]
                G = np.zeros((n_samples*(k+1), 3))
                G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
                G[:, 1] = np.ravel(idx_new, order='F')
                G[:, 2] = 1
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            t = kwargs['t']
            # compute pairwise euclidean distances
            D = pairwise_distances(X)
            D **= 2
            # sort the distance matrix D in ascending order
            dump = np.sort(D, axis=1)
            idx = np.argsort(D, axis=1)
            idx_new = idx[:, 0:k+1]
            dump_new = dump[:, 0:k+1]
            # compute the pairwise heat kernel distances
            dump_heat_kernel = np.exp(-dump_new/(2*t*t))
            G = np.zeros((n_samples*(k+1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
            G[:, 1] = np.ravel(idx_new, order='F')
            G[:, 2] = np.ravel(dump_heat_kernel, order='F')
            # build the sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

        elif kwargs['weight_mode'] == 'cosine':
            # normalize the data first
            X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
            for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
            # compute pairwise cosine distances
            D_cosine = np.dot(X, np.transpose(X))
            # sort the distance matrix D in ascending order
            dump = np.sort(-D_cosine, axis=1)
            idx = np.argsort(-D_cosine, axis=1)
            idx_new = idx[:, 0:k+1]
            dump_new = -dump[:, 0:k+1]
            G = np.zeros((n_samples*(k+1), 3))
            G[:, 0] = np.tile(np.arange(n_samples), (k+1, 1)).reshape(-1)
            G[:, 1] = np.ravel(idx_new, order='F')
            G[:, 2] = np.ravel(dump_new, order='F')
            # build the sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

    # choose supervised neighborMode
    elif kwargs['neighbor_mode'] == 'supervised':
        k = kwargs['k']
        # get true labels and the number of classes
        y = kwargs['y']
        label = np.unique(y)
        n_classes = np.unique(y).size
        # construct the weight matrix W in a fisherScore way, W_ij = 1/n_l if yi = yj = l, otherwise W_ij = 0
        if kwargs['fisher_score'] is True:
            W = lil_matrix((n_samples, n_samples))
            for i in range(n_classes):
                class_idx = (y == label[i])
                class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
                W[class_idx_all] = 1.0/np.sum(np.sum(class_idx))
            return W

        # construct the weight matrix W in a reliefF way, NH(x) and NM(x,y) denotes a set of k nearest
        # points to x with the same class as x, a different class (the class y), respectively. W_ij = 1 if i = j;
        # W_ij = 1/k if x_j \in NH(x_i); W_ij = -1/(c-1)k if x_j \in NM(x_i, y)
        if kwargs['reliefF'] is True:
            # when xj in NH(xi)
            G = np.zeros((n_samples*(k+1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                D = pairwise_distances(X[class_idx, :])
                D **= 2
                idx = np.argsort(D, axis=1)
                idx_new = idx[:, 0:k+1]
                n_smp_class = (class_idx[idx_new[:]]).size
                if len(class_idx) <= k:
                    k = len(class_idx) - 1
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class+id_now, 2] = 1.0/k
                id_now += n_smp_class
            W1 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            # when i = j, W_ij = 1
            for i in range(n_samples):
                W1[i, i] = 1
            # when x_j in NM(x_i, y)
            G = np.zeros((n_samples*k*(n_classes - 1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx1 = np.column_stack(np.where(y == label[i]))[:, 0]
                X1 = X[class_idx1, :]
                for j in range(n_classes):
                    if label[j] != label[i]:
                        class_idx2 = np.column_stack(np.where(y == label[j]))[:, 0]
                        X2 = X[class_idx2, :]
                        D = pairwise_distances(X1, X2)
                        idx = np.argsort(D, axis=1)
                        idx_new = idx[:, 0:k]
                        n_smp_class = len(class_idx1)*k
                        G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx1, (k, 1)).reshape(-1)
                        G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx2[idx_new[:]], order='F')
                        G[id_now:n_smp_class+id_now, 2] = -1.0/((n_classes-1)*k)
                        id_now += n_smp_class
            W2 = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W2) > W2
            W2 = W2 - W2.multiply(bigger) + np.transpose(W2).multiply(bigger)
            W = W1 + W2
            return W

        if kwargs['weight_mode'] == 'binary':
            if kwargs['metric'] == 'euclidean':
                G = np.zeros((n_samples*(k+1), 3))
                id_now = 0
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise euclidean distances for instances in class i
                    D = pairwise_distances(X[class_idx, :])
                    D **= 2
                    # sort the distance matrix D in ascending order for instances in class i
                    idx = np.argsort(D, axis=1)
                    idx_new = idx[:, 0:k+1]
                    n_smp_class = len(class_idx)*(k+1)
                    G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                    G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                    G[id_now:n_smp_class+id_now, 2] = 1
                    id_now += n_smp_class
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

            if kwargs['metric'] == 'cosine':
                # normalize the data first
                X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
                for i in range(n_samples):
                    X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
                G = np.zeros((n_samples*(k+1), 3))
                id_now = 0
                for i in range(n_classes):
                    class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                    # compute pairwise cosine distances for instances in class i
                    D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                    # sort the distance matrix D in descending order for instances in class i
                    idx = np.argsort(-D_cosine, axis=1)
                    idx_new = idx[:, 0:k+1]
                    n_smp_class = len(class_idx)*(k+1)
                    G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                    G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                    G[id_now:n_smp_class+id_now, 2] = 1
                    id_now += n_smp_class
                # build the sparse affinity matrix W
                W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
                bigger = np.transpose(W) > W
                W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
                return W

        elif kwargs['weight_mode'] == 'heat_kernel':
            G = np.zeros((n_samples*(k+1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                # compute pairwise cosine distances for instances in class i
                D = pairwise_distances(X[class_idx, :])
                D **= 2
                # sort the distance matrix D in ascending order for instances in class i
                dump = np.sort(D, axis=1)
                idx = np.argsort(D, axis=1)
                idx_new = idx[:, 0:k+1]
                dump_new = dump[:, 0:k+1]
                t = kwargs['t']
                # compute pairwise heat kernel distances for instances in class i
                dump_heat_kernel = np.exp(-dump_new/(2*t*t))
                n_smp_class = len(class_idx)*(k+1)
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_heat_kernel, order='F')
                id_now += n_smp_class
            # build the sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W

        elif kwargs['weight_mode'] == 'cosine':
            # normalize the data first
            X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
            for i in range(n_samples):
                X[i, :] = X[i, :]/max(1e-12, X_normalized[i])
            G = np.zeros((n_samples*(k+1), 3))
            id_now = 0
            for i in range(n_classes):
                class_idx = np.column_stack(np.where(y == label[i]))[:, 0]
                # compute pairwise cosine distances for instances in class i
                D_cosine = np.dot(X[class_idx, :], np.transpose(X[class_idx, :]))
                # sort the distance matrix D in descending order for instances in class i
                dump = np.sort(-D_cosine, axis=1)
                idx = np.argsort(-D_cosine, axis=1)
                idx_new = idx[:, 0:k+1]
                dump_new = -dump[:, 0:k+1]
                n_smp_class = len(class_idx)*(k+1)
                G[id_now:n_smp_class+id_now, 0] = np.tile(class_idx, (k+1, 1)).reshape(-1)
                G[id_now:n_smp_class+id_now, 1] = np.ravel(class_idx[idx_new[:]], order='F')
                G[id_now:n_smp_class+id_now, 2] = np.ravel(dump_new, order='F')
                id_now += n_smp_class
            # build the sparse affinity matrix W
            W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
            bigger = np.transpose(W) > W
            W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
            return W


def lap_score(df,n_selected_features, **kwargs):
    """
    This function implements the laplacian score feature selection, steps are as follows:
    1. Construct the affinity matrix W if it is not specified
    2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat)
    Input
    -----
    X: {Dataframe}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        W: {sparse matrix}, shape (n_samples, n_samples)
            input affinity matrix
    Output
    ------
    score: {numpy array}, shape (n_features,)
        laplacian score for each feature
    Reference
    ---------
    He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
    """

    if df.empty:
        return None

    X = df.values

    # if 'W' is not specified, use the default W
    if 'W' not in kwargs.keys():
        W = construct_W(X)
    # construct the affinity matrix W
    else:
        W = kwargs['W']

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000

    # compute laplacian score for all features
    score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]
    score = np.transpose(score)
    idx = np.argsort(score, 0)

    # obtain the dataset on the selected features
    selected_features_df = df.iloc[:, idx[:n_selected_features]]

    return selected_features_df


def spec(df, n_selected_features, **kwargs):
    """
    This function implements the SPEC feature selection
    Input
    -----
    df: {pandas Dataframe}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        style: {int}
            style == -1, the first feature ranking function, use all eigenvalues
            style == 0, the second feature ranking function, use all except the 1st eigenvalue
            style >= 2, the third feature ranking function, use the first k except 1st eigenvalue
        W: {sparse matrix}, shape (n_samples, n_samples}
            input affinity matrix
    Output
    ------
    w_fea: {numpy array}, shape (n_features,)
        SPEC feature score for each feature
    Reference
    ---------
    Zhao, Zheng and Liu, Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
    """

    if df.empty:
        return None

    X = df.values

    if 'style' not in kwargs:
        kwargs['style'] = 0
    if 'W' not in kwargs:
        kwargs['W'] = rbf_kernel(X, gamma=1)

    style = kwargs['style']
    W = kwargs['W']
    if type(W) is np.ndarray:
        W = csc_matrix(W)

    n_samples, n_features = X.shape

    # build the degree matrix
    X_sum = np.array(W.sum(axis=1))
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        D[i, i] = X_sum[i]

    # build the laplacian matrix
    L = D - W
    d1 = np.power(np.array(W.sum(axis=1)), -0.5)
    d1[np.isinf(d1)] = 0
    d2 = np.power(np.array(W.sum(axis=1)), 0.5)
    v = np.dot(np.diag(d2[:, 0]), np.ones(n_samples))
    v = v/np.linalg.norm(v)

    # build the normalized laplacian matrix
    L_hat = (np.matlib.repmat(d1, 1, n_samples)) * np.array(L) * np.matlib.repmat(np.transpose(d1), n_samples, 1)

    # calculate and construct spectral information
    s, U = np.linalg.eigh(L_hat)
    s = np.flipud(s)
    U = np.fliplr(U)

    # begin to select features
    w_fea = np.ones(n_features)*1000

    for i in range(n_features):
        f = X[:, i]
        F_hat = np.dot(np.diag(d2[:, 0]), f)
        l = np.linalg.norm(F_hat)
        if l < 100*np.spacing(1):
            w_fea[i] = 1000
            continue
        else:
            F_hat = F_hat/l
        a = np.array(np.dot(np.transpose(F_hat), U))
        a = np.multiply(a, a)
        a = np.transpose(a)

        # use f'Lf formulation
        if style == -1:
            w_fea[i] = np.sum(a * s)
        # using all eigenvalues except the 1st
        elif style == 0:
            a1 = a[0:n_samples-1]
            w_fea[i] = np.sum(a1 * s[0:n_samples-1])/(1-np.power(np.dot(np.transpose(F_hat), v), 2))
        # use first k except the 1st
        else:
            a1 = a[n_samples-style:n_samples-1]
            w_fea[i] = np.sum(a1 * (2-s[n_samples-style: n_samples-1]))

    if style != -1 and style != 0:
        w_fea[w_fea == 1000] = -1000

    idx_min_sort = np.argsort(w_fea, 0)
    selected_features_df = df.iloc[:,idx_min_sort[:n_selected_features]]

    return selected_features_df


def feature_ranking(score, **kwargs):
    if 'style' not in kwargs:
        kwargs['style'] = 0
    style = kwargs['style']

    # if style = -1 or 0, ranking features in descending order, the higher the score, the more important the feature is
    if style == -1 or style == 0:
        idx = np.argsort(score, 0)
        return idx[::-1]
    # if style != -1 and 0, ranking features in ascending order, the lower the score, the more important the feature is
    elif style != -1 and style != 0:
        idx = np.argsort(score, 0)
        return idx

def unsupervised_feature_selection(data,maximium_features):

    try:
        X = data.values
        W = construct_W(X,weight_mode='heat_kernel')
        data_lap = lap_score(data,maximium_features,W=W)
        data_spec = spec(data,maximium_features)

        # takes the intersection of both methods
        features_intersection_columns = list(set(data_lap) & set(data_spec))

        if not features_intersection_columns:
            print("Warning: unsupervised feature selection not found shared features, returning feature selection according to 'Spec' algorithm")
            return data_spec

        return data[features_intersection_columns]


    except Exception as e:
        raise Exception('Error in Unsupervised Feature Selection method ' + e)
