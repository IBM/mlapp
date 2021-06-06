import operator
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import ensure_float64
import statsmodels.formula.api as smf
# from skfeature.function.similarity_based import reliefF
from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold, RFE, SelectFromModel
from sklearn.linear_model import LinearRegression, LassoCV
from feature_utilities import correlate_features


def get_useful_values(features, weights, params={'threshold_weight': 0.1}):
    if not 'threshold_weight' in params:
        raise Exception('params must have threshold_weight as the minimum weight to keep')
    threshold_weight = params['threshold_weight']
    try:
        abs_weight = [abs(weight) for weight in weights]
        weight_per_feature = zip(features, abs_weight)
        useful_features = filter(lambda x: (x[1] > threshold_weight), weight_per_feature)
        return [sign_feature[0] for sign_feature in useful_features]
    except Exception as e:
        print("Failed to select useful features; ", e.message)
        return []


def select_top_correlated_features(X_df, y, params={'top_correlated': 6}):
    if not 'top_correlated' in params:
        raise Exception('params must have top_correlated as the number of features to keep')
    if not isinstance(X_df, pd.DataFrame):
        raise Exception('X_df must be of type pandas.DataFrame')
    if isinstance(y, pd.DataFrame):
        y = np.asarray(y[y.columns[0]])
        raise Warning('y was given as DataFrame, will be converted to array using the first column please be aware!!!')
    else:
        y = np.asarray(y)
    y = ensure_float64(y)
    top_correlated = params['top_correlated']
    correlation = correlate_features(X_df, y)
    sorted_correlation = np.array(sorted(correlation.items(),key=operator.itemgetter(1), reverse=True ))
    features_to_select = [feature[0] for feature in sorted_correlation[0:top_correlated]]
    return X_df[features_to_select]


def select_features_with_RFE(X, y, params={'prop':0.5}):
    if not 'prop' in params:
        raise Exception('params must have prop as the proportion of features to keep')
    if not isinstance(X, pd.DataFrame):
        raise Exception('X_df must be of type pandas.DataFrame')
    if isinstance(y, pd.DataFrame):
        y = np.asarray(y[y.columns[0]])
        raise Warning('y was given as DataFrame, will be converted to array using the first column please be aware!!!')
    else:
        y = np.asarray(y)
    y = ensure_float64(y)
    prop = params['prop']
    features = X.columns.tolist()
    trained_lin_reg = LinearRegression().fit(X, y)
    selector = RFE(trained_lin_reg, 1, step=1)
    selector = selector.fit(X, y)
    features_ranking = selector.ranking_
    features_rank = {features[f_index]: features_ranking[f_index] for f_index in range(len(features))}
    features_to_select = select_features_recursively(X, features_rank, prop)
    return X[features_to_select]


def select_features_recursively(X, feature_to_rank, prop):
    sorted_features = sorted(feature_to_rank.items(), key=operator.itemgetter(1))
    features_to_select = []
    features_counter = 0
    for potential_feature, rank in sorted_features:
        if features_counter < int(float(len(feature_to_rank.keys()) * prop)):
            should_be_added = True
            for feature in features_to_select:
                correlation = correlate_features(X[potential_feature].to_frame(), X[feature])
                if correlation[potential_feature] > 0.8:
                    should_be_added = False
                    break
            if should_be_added:
                features_to_select.append(potential_feature)
                features_counter += 1
        else:
            break
    return features_to_select


def feature_selection_AIC(X, y, params={}):
    if not isinstance(X, pd.DataFrame):
        raise Exception('X_df must be of type pandas.DataFrame')
    X_copy = X.copy()
    X_copy.rename(columns={X.columns.values[index]: 'C' + str(index) for index in range(len(X.columns.values))}, inplace=True)
    data = pd.concat([X_copy, y], axis=1, join_axes=[X_copy.index])
    remaining = set(X_copy.columns)
    selected = []
    current_score, best_new_score = 3000.0, 3000.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 0".format(y.name, ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort(reverse=True)
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score >= best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    selected = [X.columns.values[int(feature[1:len(feature)])] for feature in selected]
    return X[selected]


def select_from_model_and_lasso(X, y, params={'threshold': 0.005, 'num_of_features': 5}):
    '''
    Transformer for selecting features based on importance weights.
    :param X: Initial Dataset
    :param y: To Predict
    :param params: dictionnary with threshold and num_of_features
    threshold can be either a number - if the weight is lower than this threshold, the features 
    will be discarded -, "mean" or "median" - the weight will be compared to this metric - or -np.inf if 
    we care only about number of features 
    num_of_features: maximum number of featyres
    :return: Transformed dataframe 
    '''
    if not 'threshold' in params:
        raise Exception('params must have threshold as a parameter')
    if not 'num_of_features' in params:
        raise Exception('params must have num_of_features as a parameter')
    if not isinstance(X, pd.DataFrame):
        raise Exception('X must be of type pandas.DataFrame')
    if isinstance(y, pd.DataFrame):
        y = np.asarray(y[y.columns[0]])
        raise Warning('y was given as DataFrame, will be converted to array using the first column please be aware!!!')
    else:
        y = np.asarray(y)
    y = ensure_float64(y)
    clf = LassoCV()
    threshold = params['threshold']
    num_of_features = params['num_of_features']
    sfm = SelectFromModel(clf, threshold=threshold, max_features=num_of_features)
    sfm.fit(X, y)
    X_transform = pd.DataFrame(columns=X.columns[sfm.get_support()], data=sfm.transform(X))
    return X_transform

def select_by_relieff(X, y, params={'k':0.7, 'relieff_params': {'k':5}}):
    """
        select k% best features based on the relieff implementation
        :param X: Features dataframe
        :param y: array of floats
        :param params: dictionary with two params:
        'k' is the percent of features to leave
        'relieff_params' dictionary contains the number of neighbors to crate the graph adjacency matrix
        :return:datafram with selected features only
    """
    pass
#     if not 'k' in params:
#         raise Exception('params must have k as the percent of features to leave')
#     if not 'relieff_params' in params:
#         raise Exception('params must have k_neighbors for the relieff algorith,m')
#     if not isinstance(X, pd.DataFrame):
#         raise Exception('X must be of type pandas.DataFrame')
#     if isinstance(y, pd.DataFrame):
#         y = np.asarray(y[y.columns[0]])
#         raise Warning('y was given as DataFrame, will be converted to array using the first column please be aware!!!')
#     else:
#         y = np.asarray(y)
#     y = ensure_float64(y)
#
#     k = max(1, int(np.floor(len(X.columns) * params['k'])))
#
#     scores = reliefF.reliefF(X.values, y, **params['relieff_params'])
#
#     all_scores_excluding_nans = scores[np.isfinite(scores)]
#     all_scores_sorted = sorted(all_scores_excluding_nans, reverse=True)
#     # (k-1) index available
#     if (k - 1) < len(all_scores_sorted):
#         threshold = all_scores_sorted[k - 1]
#         relevant_columns = np.where(scores >= threshold)
#         return X[X.columns[relevant_columns]]
#     else:
#         threshold = all_scores_sorted[-1]
#         relevant_columns = np.where(scores >= threshold)
#         return X[X.columns[relevant_columns]]


def select_k_best(X, y, params={'k':0.7, 'is_regression':True}):
    '''
    select k% best features based on the sklearn implementation
    :param X: Features dataframe
    :param y: array of floats
    :param params: dictionary with two params:
    'k' is the percent of features to leave
    'is_regression' identifies whether y is continuous or classified to set the scoring function accordingly
    :return:datafram with selected features only
    '''
    if not 'k' in params:
        raise Exception('params must have k as the percent of features to leave')
    if not 'is_regression' in params:
        raise Exception('params must have is_regression to indicate the type of scoring to make')
    if not isinstance(X, pd.DataFrame):
        raise Exception('X must be of type pandas.DataFrame')
    if isinstance(y, pd.DataFrame):
        y = np.asarray(y[y.columns[0]])
        raise Warning('y was given as DataFrame, will be converted to array using the first column please be aware!!!')
    else:
        y = np.asarray(y)
    y = ensure_float64(y)

    k = max(1, int(np.floor(len(X.columns) * params['k'])))
    if params['is_regression']:
        model = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=k)
    else:
        model = feature_selection.SelectKBest(k=k)

    res = model.fit(X, y)
    all_scores = res.scores_
    all_scores_excluding_nans = res.scores_[np.isfinite(res.scores_)]
    all_scores_sorted = sorted(all_scores_excluding_nans,reverse=True)
    # (k-1) index available
    if (k - 1) < len(all_scores_sorted):
        threshold = all_scores_sorted[k - 1]
        relevant_columns = np.where(all_scores >= threshold)
        return X[X.columns[relevant_columns]]
    else:
        threshold = all_scores_sorted[-1]
        relevant_columns = np.where(all_scores >= threshold)
        return X[X.columns[relevant_columns]]


def features_selection_per_class(X, y, params={'threshold': 0.50, 'top_features': 15}):
    if not 'threshold' in params:
        raise Exception('params must have threshold as the threshold of specificity')
    if not 'top_features' in params:
        raise Exception('params must have top_features as the number of features to keep')
    if not isinstance(X, pd.DataFrame):
        raise Exception('X must be of type pandas.DataFrame')

    averages = X.mean(axis=0)
    df = X.copy()
    df['clusters'] = y
    threshold = params.get('threshold', 0.50)
    top_features = params.get('top_features', len(X.columns))
    final_features = []
    for cluster in set(df['clusters'].values):
        averages_comparison = averages.copy()
        averages_comparison = averages_comparison.to_frame()
        averages_comparison.rename(columns={averages_comparison.columns.values[0]: 'main_avg'}, inplace=True)
        selected_df = df[df['clusters'] == cluster]
        selected_df.drop('clusters', axis=1, inplace=True)
        averages_for_cluster = selected_df.mean(axis=0)
        averages_comparison['avg_for_cluster'] = averages_for_cluster
        averages_comparison.reset_index(inplace=True)
        averages_comparison['dev'] = (averages_comparison['main_avg'].apply(float) - averages_comparison[
            'avg_for_cluster']) / averages_comparison['main_avg']
        averages_comparison.sort_values(by='dev', ascending=False, inplace=True)
        averages_comparison = averages_comparison[averages_comparison['dev'] > threshold]
        if len(averages_comparison) > top_features:
            final_features += list(averages_comparison.head(top_features)['index'].values)
        else:
            final_features += list(averages_comparison['index'].values)
    return X[list(set(list(final_features)))]


def select_best_transformation(X, y, params={'max_features': 1, 'original_feature_names': []}):
    '''
    This function will select the best transformation( i.e. lead, lag, exp, etc.) according to the highest correlation with y
    Interactions(feature names containing ' x ' ) are all kept
    :param X: Initial Dataframe
    :param y: y
    :param params: dictionary with two params:
    'original_feature_names' - array of the names of the original feature before transformation,
    each transformation name must contain the original name in it, i.e. 'price_lag_1'
    'max_features' - how many correlated features from each group of transformations, default=1
    :return: dataframe with selected features only
    '''

    if not 'original_feature_names' in params:
        raise Exception('params must include original_feature_names key')
    if not 'max_features' in params:
        raise Exception('params must include max_features key')
    if not isinstance(X, pd.DataFrame):
        raise Exception('X must be of type pandas.DataFrame')

    original_feature_names = params['original_feature_names']
    max_features = params['max_features']
    interaction_cols = [curr_column for curr_column in X.columns if ' x ' in curr_column]
    interactions = X[interaction_cols]
    other_columns = [col for col in X.columns if col not in interaction_cols]
    final_selection = pd.DataFrame()

    for col in original_feature_names:
        column_transformations = [curr_column for curr_column in other_columns if col in curr_column]
        curr_features = X[column_transformations]
        top_transformation = select_top_correlated_features(curr_features, y, params={'top_correlated': max_features})
        final_selection[top_transformation.columns] = top_transformation[top_transformation.columns]

    final_selection[interaction_cols] = interactions
    return final_selection


def select_by_variance(X, y, params={'threshold': 10}):
    if not 'threshold' in params:
        raise Exception('params must have threshold as the threshold of variance')
    if not isinstance(X, pd.DataFrame):
        raise Exception('X must be of type pandas.DataFrame')

    threshold = params['threshold']
    sel = VarianceThreshold(threshold=threshold)
    sel.fit_transform(X)
    return X
