import numpy as np
import pandas as pd


def create_user_and_item_dicts_and_replace_values(df):
    # Creating mappings for replace
    dicts = {'id_to_user': dict(enumerate(df['user'].unique())), 'id_to_item': dict(enumerate(df['item'].unique()))}

    # Creating mirror mappings
    dicts['users_to_id'] = {y: x for x, y in dicts['id_to_user'].items()}
    dicts['items_to_id'] = {y: x for x, y in dicts['id_to_item'].items()}

    # Replacing user and item values in DataFrame
    df['user'] = df['user'].map(dicts['users_to_id'])
    df['item'] = df['item'].map(dicts['items_to_id'])

    return df, dicts


def pivot_table_group_by_user_and_item(df):
    return df.pivot_table(values=['quantity', 'date_min', 'date_max', 'order_id'], index=['user', 'item'], aggfunc={
        'quantity': np.sum,
        'date_min': np.min,
        'date_max': np.max,
        'order_id': lambda x: len(x.unique())
    })


def pivot_table_user_x_item(df):
    return df.pivot_table(values=['quantity', 'date_min', 'date_max', 'order_id'], index='user', columns='item', aggfunc={
        'quantity': np.sum,
        'date_min': np.min,
        'date_max': np.max,
        'order_id': lambda x: x
    })


def calc_popularity_data_frame(df):
    popularity = df.pivot_table(values='quantity', index='item', aggfunc=np.sum)
    popularity.sort_values('quantity', inplace=True, ascending=False)
    return popularity.reset_index()


def calc_recency_data_frame(dictionary):
    recency = pd.DataFrame(list(dictionary.keys())).reset_index()
    recency[0] = recency[0].apply(lambda x: len(recency) - x)
    return recency


def calc_association_rule_data_frame(df, orders_length):
    # pivoting table - listing order ids for each item
    association_rule = df.pivot_table(values='order_id', index='item', aggfunc=lambda x: list(x))
    # calculating number of baskets for each item
    association_rule['baskets_length'] = association_rule['order_id'].apply(lambda x: len(x))

    # list containing all association rules
    ar_list = []

    # dictionary containing calculated size of baskets with intersections of x and y
    baskets_containing_x_y = {}

    # looping through each item x,y combinations
    for item_id_x in list(association_rule.index):
        for item_id_y in list(association_rule.index):
            # current 'x_y', 'y_x' strings
            curr_y_x_str = str(item_id_y) + '_' + str(item_id_x)
            curr_x_y_str = str(item_id_x) + '_' + str(item_id_y)

            # checking if curr 'y_x' was calculated
            if curr_y_x_str not in baskets_containing_x_y:
                # calculating 'x_y'
                curr_x_and_y_val = len(set(association_rule.loc[item_id_x]['order_id'])
                                       .intersection(association_rule.loc[item_id_y]['order_id']))
                # setting value of 'x_y'
                baskets_containing_x_y[curr_x_y_str] = curr_x_and_y_val
            else:
                # loading value of 'y_x'
                curr_x_and_y_val = baskets_containing_x_y[curr_y_x_str]

            # calculating: P(X & Y) / (P(X) * P(Y))
            nominator = curr_x_and_y_val / orders_length
            denominator = (association_rule.loc[item_id_x]['baskets_length'] / orders_length) * \
                          (association_rule.loc[item_id_y]['baskets_length'] / orders_length)

            # checking case of 0 denominator
            if denominator == 0:
                ar_value = 0
            else:
                ar_value = nominator / denominator

            # appending new association rule to ar_list
            ar_list.append({
                'item_x': int(item_id_x),
                'item_y': int(item_id_y),
                'value': ar_value
            })

    return pd.DataFrame(ar_list)


def transform_data_from_implicit_to_explicit_ratings(data, transformation_type=0):
    # bought == 1, hasn't bought == 0
    if transformation_type == 0:
        return data.applymap(lambda x: 0 if pd.isna(x) else 1)
    # bought == 1, bought once = 0, hasn't bought == null
    if transformation_type == 1:
        return data.applymap(lambda x: np.nan if (pd.isna(x) or x == 0) else (0 if x == 1 else 1))
    # distribution transformation
    if transformation_type == 2:
        return data.apply(lambda x: scale_convert(x, np.min(x), np.max(x), c=0, d=5), axis=1)
    # log transformation
    if transformation_type == 3:
        return data.apply(lambda x: np.log1p(x))


def scale_convert(x, a, b, c=0, d=1):
    """converts values in the range [a,b] to values in the range [c,d]"""
    return np.ceil(c + (x - a) * float(d - c) / (b - a))
