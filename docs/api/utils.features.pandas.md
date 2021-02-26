<a name="mlapp.utils.features.pandas"></a>
# mlapp.utils.features.pandas

<a name="mlapp.utils.features.pandas.polynomial_features_labeled"></a>
#### polynomial\_features\_labeled

```python
polynomial_features_labeled(raw_input_df, power)
```

This is a wrapper function for sklearn's Ploynomial features, which returns the resulting power-matrix with
meaningful labels. It calls sklearn.preprocessing.PolynomialFeatures on raw_input_df with the specified power
parameter and return it's resulting array as a labeled pandas dataframe (i.e cols= ['a', 'b', 'a^2', 'axb', 'b^2']).

**Arguments**:

- `raw_input_df`: labeled pandas dataframe.
- `power`: The degree of the polynomial features (use the same power as you want entered into
PolynomialFeatures(power) directly).

**Returns**:

power-matrix with meaningful column labels.

<a name="mlapp.utils.features.pandas.lag_feature"></a>
#### lag\_feature

```python
lag_feature(feature_df, lag, dropna=False)
```

Shifts feature forward by a specified lag. first n=lag values will be None.

**Arguments**:

- `feature_df`: A pandas series.
- `lag`: int. lag size.
- `dropna`: performs dropna after shift. default-False.

**Returns**:

Transformed feature with the lag value added to the column name (i.e col_name_lag_5)..

<a name="mlapp.utils.features.pandas.lead_feature"></a>
#### lead\_feature

```python
lead_feature(feature_df, lead, dropna=False)
```

Shifts feature backward by a specified lead. last n=lead values will be None.

**Arguments**:

- `feature_df`: A pandas series.
- `lead`: int. lead size.
- `dropna`: performs dropna after shift. default-False.

**Returns**:

Transformed feature with the lead value added to the column name (i.e col_name_lead_5).

<a name="mlapp.utils.features.pandas.log_feature"></a>
#### log\_feature

```python
log_feature(feature_df)
```

Returns a natural log transformation of the feature.

**Arguments**:

- `feature_df`: A pandas series.

**Returns**:

Transformed feature with 'log' added to the column name (i.e col_name_log).

<a name="mlapp.utils.features.pandas.exponent_feature"></a>
#### exponent\_feature

```python
exponent_feature(feature_df)
```

Returns a natural exponent transformation of the feature.

**Arguments**:

- `feature_df`: A pandas series.

**Returns**:

Transformed feature with 'exp' added to the column name (i.e col_name_exp).

<a name="mlapp.utils.features.pandas.power_feature"></a>
#### power\_feature

```python
power_feature(feature_df, power)
```

Returns the feature raised to the power specified.

**Arguments**:

- `feature_df`: A pandas series.
- `power`: int.

**Returns**:

Transformed feature with power value added to the column name (i.e col_name_pow_5).

<a name="mlapp.utils.features.pandas.sqrt_feature"></a>
#### sqrt\_feature

```python
sqrt_feature(feature_df)
```

Returns a square root transformation of the feature.

**Arguments**:

- `feature_df`: A pandas series.

**Returns**:

Transformed feature with 'sqrt' added to the column name (i.e col_name_sqrt).

<a name="mlapp.utils.features.pandas.inverse_feature"></a>
#### inverse\_feature

```python
inverse_feature(feature_df)
```

Returns the inverse (1/x) transformation of the feature.

**Arguments**:

- `feature_df`: A pandas series.

**Returns**:

Transformed feature with 'inverse' added to the column name (i.e col_name_inverse).

<a name="mlapp.utils.features.pandas.interact_features"></a>
#### interact\_features

```python
interact_features(feature_df, interact_list, drop_original_columns=True)
```

This function create interactions between pairs of features

**Arguments**:

- `feature_df`: a pandas dataframe
- `interact_list`: list of lists or tuples with two strings each, representing columns to be interacted.
- `drop_original_columns`: (bool) if set to True, columns to be interacted will be droped from the dataframe. note-
if set to true a column cannot appear in more then one interaction pair.

**Returns**:

DataFrame with the interactions columns, without the original columns.

<a name="mlapp.utils.features.pandas.extend_dataframe"></a>
#### extend\_dataframe

```python
extend_dataframe(df, y_name_col=None, index_col=None, lead_order=3, lag_order=3, power_order=2, log=True, exp=True, sqrt=True, poly_degree=2, dropna=False, fillna_value=0, inverse=True)
```

Create a new dataframe with transformed features added as new columns.

**Arguments**:

- `df`: initial dataframe
- `y_name_col`: y column name. this column will not be extended.
- `index_col`: index column/s (string or list of strings). Any column that should NOT be extended, can be listed here.
- `lead_order`: shifts the feature values back. Extended dataframe will include all leads from 1 to lead_order specified.
- `lag_order`: shifts the feature values forward. Extended dataframe will include all lags from 1 to lead_order specified.
- `power_order`: Extended dataframe will include all powers from 1 to power_order specified.
- `log`: perform natural log transformation. Default=True.
- `exp`: perform natural exponent transformation. Default=True.
- `sqrt`: perform square root tranformation. Default=True.
- `poly_degree`: the highest order polynomial for the transformation. 0=no polynomial transformation.
- `dropna`: default True.
- `fillna_value`: default True.
- `inverse`: perform inverse transformation (1/x). default=True.

**Returns**:

extended dataframe.

<a name="mlapp.utils.features.pandas.calc_t_values"></a>
#### calc\_t\_values

```python
calc_t_values(X, y, y_hat, coefficients)
```

p-values for regerssion assets
calc t values: t(b_i) = b_i / SE(b_i)
Where b_i is the beta (coefficient) of x_i
and SE(b_i) is the standard error of the coefficient
@param X: Dataframe or Series
@param y: Dataframe or Series
@param y_hat: Dataframe or Series
@param coefficients: list or ndarray
@return: numpay array

<a name="mlapp.utils.features.pandas.calc_p_values"></a>
#### calc\_p\_values

```python
calc_p_values(X, y, y_hat, coefficients)
```

p-values for regression assets. Calc p values from t values
@param X: pd.DataFrame or pd.Series
@param y: pd.DataFrame or pd.Series
@param y_hat: pd.DataFrame or pd.Series
@param coefficients: list or ndarray
@return: NumPy array

