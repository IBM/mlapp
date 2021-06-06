import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar
import datetime

def remove_nulls_and_reindex(data, index_cols_names, time_frequency):
    """
    :param data: data to remove nulls from
    :param index_cols_names: column names in data
    :param time_frequency: time frequency of datetime index
    :return:
    """
    df = data.copy()
    df = df.dropna(axis=0)
    df['date_orig'] = df['date']
    df['date'] = pd.DatetimeIndex(start=df['date'][0], periods=len(df),
                                  freq=time_frequency)
    df = df.reset_index().drop(labels={'index'}, axis=1)
    index_cols_names = [k for k in index_cols_names]
    index_cols_names.append('date_orig')
    return df, index_cols_names


def remove_null_values(df, variable_to_predict):
    """
    :param df: DataFrame containing data
    :param variable_to_predict: column name to check nulls
    :return:
    """
    null_idx = df[variable_to_predict].isnull()
    df = df[~null_idx]
    return df


def manipulate_dataset(X, features, transformations, output_transformation=None, y=[], date_selector=[],
                       seasonality=False):
    if len(date_selector) > 0:
        X_df, y_df = _date_selection(X, y, date_selector)
    else:
        X_df = X.copy()
        y_df = y

    for index_feature, feature in enumerate(features):
        X_df[feature] = X_df[feature].apply(transformations[index_feature])

    if isinstance(y, pd.DataFrame):
        if output_transformation:
            y_df[y_df.columns[0]] = y_df[y_df.columns[0]].apply(output_transformation)
        if seasonality:
            y, seasonalities = _deseasone_y(y_df)
            return X_df, y_df, y, seasonalities
        else:
            return X_df, y_df
    return X_df

def _date_selection(X_df, y_df, date_selector):
    X = X_df.copy()
    X = X.ix[(X_df.index >= date_selector[0]) & (X.index <= date_selector[len(date_selector) - 1])]
    if isinstance(y_df, pd.DataFrame):
        y = y_df.copy()
        y = y.ix[(y.index >= date_selector[0]) & (y.index <= date_selector[len(date_selector) - 1])]
    else:
        y = y_df
    return X, y


def _deseasone_y(y_df):
    df = y_df.copy()
    df.index = df.index.to_datetime()
    res = seasonal_decompose(df, freq=12)
    seasonalities = res.seasonal
    df[df.columns[0]] = df[df.columns[0]] - seasonalities[seasonalities.columns[0]]
    return df, seasonalities


def add_months(source_date, months):
    month = source_date.month - 1 + months
    year = int(source_date.year + month / 12)
    month = month % 12 + 1
    day = min(source_date.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def calc_cosine_similarity(df, v, prev_df=None):
    """
    Calculating similarities
    :param df: DataFrame with data of ratings: users-items
    :param v: Normalized vector (L2) of df
    :param prev_df: DataFrame with data of ratings from a previous calculation
    :return: return DataFrame with the calculated Cosine Similarity
    """
    if prev_df is None:
        prev_df = df
    cs = df.apply(lambda x: np.divide(np.matmul(x, prev_df), v[x.name] * v))  # Cosine Similarity
    return cs
