import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold


def split_train_test(data, timestamps=None, users_percent=0.25,
                     test_start_date=None, items_for_test=1, second_best_item=False, seed=None):
    """
    Splits data into train and test sets
    :param data: user-item ratings DataFrame
    :param timestamps: timestamp of ratings
    :param users_percent: percent of users used for test
    :param test_start_date: items taken for test from this date onwards (requires timestamps)
    :param items_for_test: number of items to use for test in each user
    :param second_best_item: whether to use second best item of user for test always
    :param seed: seed for randomness
    :return: train, test
    """
    # DataFrames to return
    train_set, test_set = data.copy(), pd.DataFrame().reindex_like(data)
    # calculate number of users for test
    num_of_users_for_test = int(len(data.index) * users_percent)
    # users for test set with more than (len(items) * items_percent) rankings
    if timestamps is None or test_start_date is None:
        users = np.random.RandomState(seed=seed).permutation(
            list(data.index[data.count(axis=1) > items_for_test]))[0:num_of_users_for_test]
    else:
        users = np.random.RandomState(seed=seed).permutation(
            list(timestamps['last'].index[np.logical_and(
                timestamps['last'].max(axis=1) >= test_start_date,
                timestamps['last'].count(axis=1) > items_for_test)])
        )[0:num_of_users_for_test]

    counter = 0
    unavailable = 0
    for user in users:
        # no timestamps - picking indices randomly
        if timestamps is None or test_start_date is None:
            # choosing items randomly
            if not second_best_item or items_for_test != 1:
                indices_for_test = np.random.RandomState(
                    seed=None if seed is None else (seed + counter)).permutation(
                    data.loc[user].dropna().index)[0:items_for_test]
            # second best item
            else:
                indices_for_test = np.array(data.loc[user].nlargest(2).index)[1:2]

        # timestamps available - picking most recent indices
        else:
            max_value = np.max(timestamps['last'].loc[user])
            max_indices = np.array(timestamps['last'].loc[user][timestamps['last'].loc[user] == max_value].index)
            matching_min_indices = np.array(timestamps['first'].loc[user].iloc[max_indices][
                timestamps['first'].loc[user].iloc[max_indices] == max_value].index)
            if len(matching_min_indices) > 0:
                indices_for_test = np.random.RandomState(
                    seed=None if seed is None else (seed + counter)).permutation(
                    matching_min_indices)[0:items_for_test]
            else:
                unavailable += 1
                counter += 1
                continue

        # removing data from train set and adding it to test set
        train_set.loc[user, indices_for_test] = np.nan
        test_set.loc[user, indices_for_test] = data.loc[user, indices_for_test]
        counter += 1
    return train_set, test_set


def cv_splits(data, timestamps=None, n_splits=5, random_state=None,
              test_start_date=None, items_for_test=1, second_best_item=False):
    """
    Splits data into cross validation splits of train and test
    :param data: user-item ratings DataFrame
    :param timestamps: containing all timestamps of last purchases
    :param n_splits: number of splits of cross validation
    :param random_state: keeping split constant between different runs
    :param test_start_date: items taken for test from this date onwards (requires timestamps)
    :param items_for_test: number of items to use for test in each user
    :param second_best_item: whether to use second best item of user for test always
    :return: [[train_set_1, test_set_1], [train_set_2, test_set_2], ...]
    """
    splits = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kf.get_n_splits(np.arange(len(data.index)))
    for train_index, test_index in kf.split(data.index,):
        # init values
        train_set, test_set, users_num = data.copy(), pd.DataFrame().reindex_like(data), len(test_index)
        unavailable = 0
        # looping through test indexes
        for i in test_index:
            # skip users with 1 rating only
            if data.iloc[i].count() <= items_for_test:
                continue
            # choosing item by most recent
            if timestamps is not None and test_start_date is not None:
                non_null_indices = np.array(data.iloc[i][pd.notnull(data.iloc[i])].index)
                max_value = np.max(timestamps['last'].iloc[i].iloc[non_null_indices])
                max_indices = np.array(timestamps['last'].iloc[i].iloc[non_null_indices][
                        timestamps['last'].iloc[i].iloc[non_null_indices] == max_value].index)
                matching_min_indices = np.array(timestamps['first'].iloc[i].iloc[max_indices][
                                                    timestamps['first'].iloc[i].iloc[max_indices] == max_value].index)
                if len(matching_min_indices) > 0 and max_value >= test_start_date:
                    test_indices = np.random.RandomState(
                        seed=None if random_state is None else (random_state + i)).permutation(
                        matching_min_indices)[0:items_for_test]
                else:
                    unavailable += 1
                    continue
            else:
                # choosing items randomly
                if not second_best_item or items_for_test != 1:
                    test_indices = np.random.RandomState(
                        seed=None if random_state is None else (random_state + i)).permutation(
                        data.iloc[i].dropna().index)[0:items_for_test]
                # second best item
                else:
                    test_indices = np.array(data.iloc[i].nlargest(2).index)[1:2]
            # removing data from train set and adding it to test set
            train_set.iloc[i, test_indices] = np.nan
            test_set.iloc[i, test_indices] = data.iloc[i, test_indices]
        splits.append([train_set, test_set])

    return splits


def score_function(func, predictions, train_set, test_set, black_list=None, **kwargs):
    transformed_predictions = transform_predictions(predictions, len(test_set), transform_type=func)
    if func == 'ranking':
        return calculate_ranking(transformed_predictions, train_set, test_set, black_list)
    else:
        return calculate_score(transformed_predictions, test_set, score_func=func, **kwargs)


def calculate_ranking(rankings, train_set, test_set, black_list=None):
    """
    Calculating accuracy per user-item ratings (implicit data)
    :param rankings: data containing predicted ratings
    :param train_set: train data containing actual implicit data such as purchases
    :param test_set: test data containing actual implicit data such as purchases
    :param black_list: items not to recommend
    :return: accuracy scoring: percent of successful bucket rating
    """
    if black_list is None:
        black_list = []

    helper_df = pd.DataFrame()

    # indices of all null item per user
    helper_df['null_items'] = train_set.apply(
        lambda x: [idx for idx, val in enumerate(x) if pd.isnull(val) and idx not in black_list], axis=1)
    helper_df['items_ranking'] = rankings

    # index of test item per user
    helper_df['test_item'] = test_set.apply(lambda x: [idx for idx, val in enumerate(x) if pd.notnull(val)], axis=1)
    # TODO: this test_item takes into account only one item for test (x[0]), check this row in case of multiple items
    helper_df['test_item'] = helper_df['test_item'].apply(lambda x: np.nan if len(x) == 0 else int(x[0]))

    # choose rows only of test users
    helper_df = helper_df[pd.notnull(helper_df['test_item'])]

    # random evaluation
    # run_random_model_ranking_evaluation(helper_df)

    # TODO: score takes into account only one item for test, check this in case of multiple test items
    # calculate score - percent of test item rank higher than non-purchased item rank
    helper_df['auc_score'] = helper_df.apply(
        lambda x: 1 if len(x['null_items']) <= 1 else np.sum(
            [1 for i in x['null_items']
             if x['test_item'] != i and x['items_ranking'].index(x['test_item']) < x['items_ranking'].index(i)
             ])/(len(x['null_items']) - 1), axis=1)

    # calculate score - 1/0 whether test item is inside top bucket_size
    for bucket_size in range(1, 4):
        helper_df['top_' + str(bucket_size)] = helper_df.apply(
            lambda x: x['test_item'] in [i for i in x['items_ranking'] if i in x['null_items']][0:bucket_size],
            axis=1)

    # scores
    scores = {
        'top_' + str(bucket_size): float(helper_df['top_' + str(bucket_size)].sum())/len(helper_df)
        for bucket_size in range(1, 4)
    }
    scores.update({'auc': float(helper_df['auc_score'].sum()) / len(helper_df)})

    return scores


def transform_predictions(predictions, num_of_users, transform_type='ranking'):
    """
    transforming predictions to ranking sequences
    :param predictions: DataFrame/Series/list of scores for each user-item
    :param num_of_users: number of users
    :param transform_type: transformation type (ranking/score)
    :return: Series which contains ranking order for each user
    """
    transform_types = {
        'ranking': 0,
        'score': 1
    }

    # predictions == `Users x Items` DataFrame
    if isinstance(predictions, pd.DataFrame):
        # ordering ranking by value for each user
        if transform_type == 'ranking':
            output = predictions.apply(
                lambda x: [idx for idx, val in sorted(enumerate(x), key=lambda y: y[1], reverse=True)], axis=1)
        else:
            # transform_type == 'score'
            output = predictions.apply(lambda x: [val for val in x], axis=1)
    # predictions == Series
    elif isinstance(predictions, pd.Series):
        # ordering is already calculated for each user
        if transform_type == 'ranking':
            output = predictions.apply(
                lambda x: [int(idx) for idx, val in sorted(x.items(), key=lambda y: y[1], reverse=True)])
        else:
            # transform_type == 'score'
            output = predictions.apply(lambda x: [val for idx, val in sorted(x.items(), key=lambda y: int(y[0]))])
    # predictions == list
    elif isinstance(predictions, list):
        # order is the list hence duplicating same order for each user
        values = list(predictions[transform_types[transform_type]].values())
        output = [values for _ in range(num_of_users)]
    else:
        raise Exception("Error in `predictions` argument: unsupported format.")

    return output


def calculate_score(predictions, test_set, score_func, binary_classification=False):
    """
    Calculating accuracy per user-item ratings (explicit data)
    :param predictions: data containing predicted ratings
    :param test_set: data containing actual ratings
    :param score_func: score function to use
    :param binary_classification:  whether data is binary or not
    :return: accuracy scoring: auc, rmse, mae
    """
    # select relevant values
    relevant_pred = predictions[pd.notnull(test_set)]
    relevant_actuals = test_set

    # convert matrix to vectors
    if hasattr(relevant_pred, 'values'):
        relevant_pred = relevant_pred.values.ravel()
    if hasattr(relevant_actuals, 'values'):
        relevant_actuals = relevant_actuals.values.ravel()

    # drop nulls
    relevant_pred = relevant_pred[pd.notnull(relevant_pred)]
    relevant_actuals = relevant_actuals[pd.notnull(relevant_actuals)]

    # round to 1/0 or ceiling
    if binary_classification:
        relevant_pred = [0 if x < 0.5 else 1 for x in relevant_pred]
    else:
        relevant_pred = np.ceil(relevant_pred)

    calculation = relevant_actuals - relevant_pred

    if score_func == 'rmse':
        return np.sqrt(np.mean(calculation ** 2))
    elif score_func == 'mase':
        return np.mean(np.abs(calculation))
    elif score_func == 'auc':
        return roc_auc_score(relevant_actuals, relevant_pred)
    elif score_func == 'f1':
        return f1_score(relevant_actuals, relevant_pred, average='macro')


def run_random_model_ranking_evaluation(df):
    for run_num in range(10):
        print("> Random test run:", run_num)
        df['random_rankings'] = df['items_ranking'].apply(lambda x: list(np.random.permutation(x)))

        # top bucket size (1-10)
        for bucket_size in range(1, 4):
            df['random_score_top_' + str(bucket_size)] = df.apply(
                lambda x: x['test_item'] in [i for i in x['random_rankings'] if i in x['null_items']][0:bucket_size],
                axis=1)
            random_score_top = float(df['random_score_top_' + str(bucket_size)].sum()) / len(df)
            print("> Top ", bucket_size, ": ", random_score_top)

        # auc
        df['random_score_auc'] = df.apply(
            lambda x: 1 if len(x['null_items']) <= 1 else np.sum(
                [1 for i in x['null_items']
                 if x['test_item'] != i and x['random_rankings'].index(x['test_item']) < x['random_rankings'].index(i)
                 ]) / (len(x['null_items']) - 1), axis=1)
        random_score_auc = float(df['random_score_auc'].sum()) / len(df)
        print("> AUC: ", random_score_auc)
