from pyspark.ml.feature import Bucketizer
import pyspark.sql.functions as F


def spark_cut(df, col_name, bins, labels):
    """
    Turns a continuous variable into categorical.
    :param df: a spark dataframe
    :param col_name: the continuous column to be categorized.
    :param bins: lower and upper bounds. must be sorted ascending and encompass the col entire range.
    :param labels: labels for each category. should be len(bins)-1
    :return: a spark dataframe with the specified column binned and labeled as specified.
    """
    bucketizer = Bucketizer(splits=bins,
                            inputCol=col_name,
                            outputCol=col_name + '_binned')

    df = bucketizer.transform(df)
    label_array = F.array(*(F.lit(label) for label in labels))
    df = df.withColumn(col_name, label_array.getItem(F.col(col_name + '_binned').cast('integer')))
    df = df.drop(col_name + '_binned')
    return df


def spark_dummies(data, columns=None, drop_first=False):
    """
    returns a new dataframe with a hot vector column for each unique value for each column specified in columns.
    the column itself will be deleted. If no columns are provided, all columns with less then 10 distinct values
    will be converted.
    :param data: a spark dataframe
    :param columns: string or array of strings specifying columns to transform.
    :param drop_first: default=False. If set to True, first column will be removed to avoid multicollinearity.
    :return:  new dataframe with a column for each unique value in the specified columns
    """

    if columns is None:
        columns = [c for c in data.columns if data.select(c).distinct().count() < 10]
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:

        categories = list(map(lambda x: x[0], data.select(col).distinct().collect()))

        dummies = [F.when(F.col(col) == category, 1).otherwise(0).alias(f"{col}_{category}") for
                   category in categories]
        if drop_first:
            dummies.pop(0)
        data = data.select(data.columns + dummies)
        data = data.drop(col)
    return data


def spark_select_dummies(data, column_prefix, targets, combine_to_other=False):
    """
    Use to select dummy variables to keep. values not kept will either be deleted or minimized to a single "other"
    column.
    :param data: a spark dataframe
    :param column_prefix: string. the original column name that was converted to dummies.
    :param targets: String or array of strings. The values to be preserved.
    :param combine_to_other: Default=False. If set to True, columns corresponding to values NOT in target will be
    minimized to a single "colName_other" column.
    :return: new dataframe with non target value columns deleted or minimized.
    """
    if isinstance(targets, str):
        targets = [targets]
    cols_to_combine = []
    for col in data.columns:
        if col.startswith(column_prefix) and col[len(column_prefix)+1:] not in targets:
            if combine_to_other:
                cols_to_combine.append(col)
            else:
                data = data.drop(col)
    if len(cols_to_combine)>0:
        data = data.withColumn(f'{column_prefix}_temp', sum(data[col] for col in cols_to_combine))
        data = data.withColumn(f'{column_prefix}_other', F.when(F.col(f'{column_prefix}_temp') > 0, 1).otherwise(0))
        data = data.drop(f'{column_prefix}_temp')
        data = data.select([c for c in data.columns if c not in cols_to_combine])
    return data
