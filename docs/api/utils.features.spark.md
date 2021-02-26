<a name="mlapp.utils.features.spark"></a>
# mlapp.utils.features.spark

<a name="mlapp.utils.features.spark.spark_cut"></a>
#### spark\_cut

```python
spark_cut(df, col_name, bins, labels)
```

Turns a continuous variable into categorical.

**Arguments**:

- `df`: a spark dataframe
- `col_name`: the continuous column to be categorized.
- `bins`: lower and upper bounds. must be sorted ascending and encompass the col entire range.
- `labels`: labels for each category. should be len(bins)-1

**Returns**:

a spark dataframe with the specified column binned and labeled as specified.

<a name="mlapp.utils.features.spark.spark_dummies"></a>
#### spark\_dummies

```python
spark_dummies(data, columns=None, drop_first=False)
```

returns a new dataframe with a hot vector column for each unique value for each column specified in columns.
the column itself will be deleted. If no columns are provided, all columns with less then 10 distinct values
will be converted.

**Arguments**:

- `data`: a spark dataframe
- `columns`: string or array of strings specifying columns to transform.
- `drop_first`: default=False. If set to True, first column will be removed to avoid multicollinearity.

**Returns**:

new dataframe with a column for each unique value in the specified columns

<a name="mlapp.utils.features.spark.spark_select_dummies"></a>
#### spark\_select\_dummies

```python
spark_select_dummies(data, column_prefix, targets, combine_to_other=False)
```

Use to select dummy variables to keep. values not kept will either be deleted or minimized to a single "other"
column.

**Arguments**:

- `data`: a spark dataframe
- `column_prefix`: string. the original column name that was converted to dummies.
- `targets`: String or array of strings. The values to be preserved.
- `combine_to_other`: Default=False. If set to True, columns corresponding to values NOT in target will be
minimized to a single "colName_other" column.

**Returns**:

new dataframe with non target value columns deleted or minimized.

