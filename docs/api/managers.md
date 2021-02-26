<a name="mlapp.managers"></a>
# mlapp.managers

<a name="mlapp.managers.DataManager"></a>
## DataManager

<a name="mlapp.managers.DataManager.load_train_data"></a>
#### load\_train\_data

```python
@pipeline
load_train_data(*args)
```

Write your own logic to load your train data.

**Returns**:

data object to be passed to next pipeline step

<a name="mlapp.managers.DataManager.clean_train_data"></a>
#### clean\_train\_data

```python
@pipeline
clean_train_data(data)
```

Write your own logic to clean your train data.

**Arguments**:

- `data`: data object received from previous pipeline step

**Returns**:

data object to be passed to next pipeline step

<a name="mlapp.managers.DataManager.transform_train_data"></a>
#### transform\_train\_data

```python
@pipeline
transform_train_data(data)
```

Write your own logic to transform your train data.

**Arguments**:

- `data`: data object received from previous pipeline step

**Returns**:

data object to be passed to next pipeline step

<a name="mlapp.managers.DataManager.load_forecast_data"></a>
#### load\_forecast\_data

```python
@pipeline
load_forecast_data(*args)
```

Write your own logic to load your forecast data.

**Returns**:

data object to be passed to next pipeline step

<a name="mlapp.managers.DataManager.clean_forecast_data"></a>
#### clean\_forecast\_data

```python
@pipeline
clean_forecast_data(data)
```

Write your own logic to clean your forecast data.

**Arguments**:

- `data`: data object received from previous pipeline step

**Returns**:

data object to be passed to next pipeline step

<a name="mlapp.managers.DataManager.transform_forecast_data"></a>
#### transform\_forecast\_data

```python
@pipeline
transform_forecast_data(data)
```

Write your own logic to transform your forecast data.

**Arguments**:

- `data`: data object received from previous pipeline step

**Returns**:

data object to be passed to next pipeline step

<a name="mlapp.managers.DataManager.cache_features"></a>
#### cache\_features

```python
@pipeline
cache_features(df)
```

Used to save the features data frame. It can later be retrieved using get_features().

**Arguments**:

- `df`: Dataframe

**Returns**:

None

<a name="mlapp.managers.DataManager.load_features"></a>
#### load\_features

```python
@pipeline
load_features()
```

Returns the dataframe "features" from input manager.

**Returns**:

df

<a name="mlapp.managers.DataManager.get_input_from_predecessor_job"></a>
#### get\_input\_from\_predecessor\_job

```python
get_input_from_predecessor_job(key=None)
```

Returns output from predecessor job "features" from input manager (in a [flow setting](/advanced-topics/part-2-flow)).

**Arguments**:

- `key`: key name of value

**Returns**:

Any object.

<a name="mlapp.managers.ModelManager"></a>
## ModelManager

<a name="mlapp.managers.ModelManager.train_model"></a>
#### train\_model

```python
@pipeline
train_model(data)
```

Write your own logic to train your model.

**Arguments**:

- `data`: data object received from previous pipeline step

**Returns**:

None

<a name="mlapp.managers.ModelManager.forecast"></a>
#### forecast

```python
@pipeline
forecast(data)
```

Write your own logic for forecast.

**Arguments**:

- `data`: data object received from previous pipeline step

**Returns**:

None

<a name="mlapp.managers.ModelManager.refit"></a>
#### refit

```python
@pipeline
refit(data)
```

Write your own logic for refit.

**Arguments**:

- `data`: data object received from previous pipeline step

**Returns**:

None

<a name="mlapp.managers.ModelManager.add_predictions"></a>
#### add\_predictions

```python
add_predictions(primary_keys_columns, y_hat, y=pd.Series(), prediction_type='TRAIN')
```

Creates a prediction dataframe and saves it to storage.

**Arguments**:

- `primary_keys_columns`: array-like. shared index for y_hat and y_true.
- `y_hat`: Series / array-like. predictions.
- `y`: Series / array-like. True values.
- `prediction_type`: String. (i.e TRAIN, TEST, FORECAST)

**Returns**:

None.

<a name="mlapp.managers.shared_functions"></a>
## Shared Functions

<a name="mlapp.managers.shared_functions.save_metadata"></a>
#### save\_metadata

```python
save_metadata(key, value)
```

Saves a metadata value to storage.

**Arguments**:

- `key`: String.
- `value`: metadata includes all "json serializable" objects (i.e string, int, dictionaries, list and tuples)

**Returns**:

None

<a name="mlapp.managers.shared_functions.get_metadata"></a>
#### get\_metadata

```python
get_metadata(key, default_value=None)
```

returns metadata value given a key.

**Arguments**:

- `key`: String.
- `default_value`: any object, default is None.

**Returns**:

metadata value.

<a name="mlapp.managers.shared_functions.save_object"></a>
#### save\_object

```python
save_object(obj_name, obj_value, obj_type='pkl')
```

Saves objects to storage.

**Arguments**:

- `obj_name`: String.
- `obj_value`: Obj.
- `obj_type`: String. One of MLApp supported file types: 'pkl', 'pyspark', 'tensorflow', 'keras', 'pytorch'. Default='pkl'.

**Returns**:

None

<a name="mlapp.managers.shared_functions.get_object"></a>
#### get\_object

```python
get_object(obj_name, default_value=None)
```

Returns an object given a key.

**Arguments**:

- `obj_name`: String.
- `default_value`: String.

**Returns**:

Object

<a name="mlapp.managers.shared_functions.save_images"></a>
#### save\_images

```python
save_images(images)
```

Saves images to storage.

**Arguments**:

- `images`: dictionary of matplotlib/pyplot figures. Keys will be used as image names.

**Returns**:

None.

<a name="mlapp.managers.shared_functions.save_image"></a>
#### save\_image

```python
save_image(image_name, image)
```

Saves image to storage.

**Arguments**:

- `image_name`: String.
- `image`: matplotlib/pyplot figure.

**Returns**:

None

<a name="mlapp.managers.shared_functions.get_dataframe"></a>
#### get\_dataframe

```python
get_dataframe(key)
```

**Arguments**:

- `key`: String

**Returns**:

the data frame

<a name="mlapp.managers.shared_functions.save_dataframe"></a>
#### save\_dataframe

```python
save_dataframe(key, value, to_table=None)
```

Save a data frame to storage.

**Arguments**:

- `key`: String
- `value`: a DataFrame.
- `to_table`: String. Database table name. default: None.

**Returns**:

None

<a name="mlapp.managers.shared_functions.save_automl_result"></a>
#### save\_automl_result

```python
save_automl_result(self, results: AutoMLResults, obj_type='pkl')
```

Saves an AutoMLResults object.

**Arguments**:

- `results`: AutoMLResults object
- `obj_type`: type of framework, Default: 'pkl'. Supports: 'pkl', 'pyspark', 'tensorflow', 'keras', 'pytorch'.

**Returns**:

None

<a name="mlapp.managers.shared_functions.get_automl_result"></a>
#### get\_automl_result

```python
get_automl_result(self) -> AutoMLResults
```

Returns an AutoMLResult object that was saved in a previous run

**Returns**:

AutoMLResult object