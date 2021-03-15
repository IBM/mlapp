# Part 6 - Improving the Asset Using MLApp's Utilities

At this point, we are ready to make some upgrade to the classifier you built. In this part you will see what can be done to quickly improve the classifier. MLApp comes with a library of utilities that we will be using now to rapidly add some important capabilities.

## Improving the Data Manager

First, open the file **assets/crash_course/crash_course_data_manager.py**.

You may have noticed that _transform_train_data_ and _transform_forecast_data_ are identical. That is because the same transformations are needed in both the train and forecasting pipelines.

In order to reduce code duplication, let's add an **inner function**, which is just a function that we will call from inside the _transform_train_data_ and _transform_forecast_data_ methods.

Add an inner function called __transform_data_ and copy the transformation logic from the previous exercise:

```python
def _transform_data(self, data):
    # apply same transformations
    features_to_log = self.data_settings.get('features_to_log', [])
    for feature in features_to_log:
        data[feature] = data[feature].apply(np.log)
    return data
```

Now, call this function from your _transform_train_data_ and _transform_forecast_data_ functions:

```python
@pipeline
def transform_train_data(self, data):
    # apply transformations
    data = self._transform_data(data)
    return data
```

```python
@pipeline
def transform_forecast_data(self, data):
    # apply transformations
    data = self._transform_data(data)
    return data
```

You are now ready to add some improvements to the transformation logic, without having to duplicate your code between the train and forecast pipelines.

MLApp comes with a library of [utilities](/api/utils.automl) that can be used in your data and model managers to improve your model.

As a first example, we will use the [extend_dataframe](/api/utils.features.pandas/#extend_dataframe) method to automatically generate many different feature transformations.

First, add a new line at the top of the data manager to import the _extend_dataframe_ method from the features_utilities library:

```python
from mlapp.utils.features.pandas import extend_dataframe
```

[extend_dataframe](/api/utils.features.pandas/#extend_dataframe) returns a new dataframe that includes transformed features like interactions, inverse and others.

To use it, modify your `_transform_data` inner function as shown below (notice we removed the manual log transformation, as it is now redundant):

```python
def _transform_data(self, data):
    # extend data frame
    conf_extend_dataframe = self.data_settings.get('conf_extend_dataframe', {})
    data = extend_dataframe(
        data,
        y_name_col=self.model_settings.get('target'),
        lead_order=conf_extend_dataframe.get('lead', 0),
        lag_order=conf_extend_dataframe.get('lag', 0),
        power_order=conf_extend_dataframe.get('power', 0),
        log=conf_extend_dataframe.get('log', False),
        exp=conf_extend_dataframe.get('exp', False),
        sqrt=conf_extend_dataframe.get('sqrt', False),
        poly_degree=conf_extend_dataframe.get('interactions', 0),
        inverse=conf_extend_dataframe.get('inverse', False)
    )
    return data
```

!!! note
    We are setting the parameters of [extend_dataframe](/api/utils.features.pandas/#extend_dataframe) via injected configuration options.
    
Lets modify the base train config to support this:
```json
"data_settings": {
    "file_path": "data/glass.csv",
    "conf_extend_dataframe": {
        "log": true,
        "exp": true,
        "interactions": 2
    }
},
```

In this example, we are using this method to create log and exponential features as well as pairwise interaction effects.

After making so many transformations to the data, we may want to store the transformed dataset and cache it for later use. This can easily be done by calling _**self.add_dataframe**_ function at the end of **_transform_train_data_**:

```python
@pipeline
def transform_train_data(self, data):
    data = self._transform_data(data)
    self.save_dataframe('features', data)
    return data
```


## Improving the Model Manager

Now let's make some upgrades to the model manager.

First, open the file **assets/crash_course/crash_course_model_manager.py** and add a new line at the top of the file to import the [AutoML Utility](/api/utils.automl/#automlpandas).

```python
from mlapp.utils.automl import AutoMLPandas
```

Lets use the `AutoMLPandas` to improve the _train_model_ function:

```python
@pipeline
def train_model(self, data):
    # splitting X and y
    target = self.model_settings['target']
    y = data[target]
    X = data.drop(target, axis=1)

    # split the data to test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.model_settings.get('train_percent'))

    # run auto ml
    result = AutoMLPandas('multi_class', **self.model_settings.get('auto_ml', {}))\
        .run(X_train, y_train, X_test, y_test)

    # print report
    result.print_report(full=False)

    # save results
    self.save_automl_result(result)
```

Once we use the [AutoML Utility](/api/utils.automl/#automlpandas) we iterate through many **multi_class** classification models, running a search on hyper parameters we set up, and return the best model based on f1-score. More comprehensive model performance scores are also calculated, such as Jaccard, precision and recall. At this example we are using the default settings but they can be [modified](/api/utils.automl/#automlsettings). 

Just like before, we will update the base configuration to support these new features:

```json
"model_settings": {
    "train_percent": 0.8,
    "target": "Type",
    "auto_ml": {
        "models": ["RandomForest", "ExtraTreeClassifier"],
        "hyper_params": {
            "RandomForest": {
                "max_depth": "range(1, 3)",
                "n_estimators": [400, 700, 1000]
            },
            "ExtraTreeClassifier": {
                "n_estimators": [10, 50, 200],
                "min_samples_split": [2, 10]
            }
        },
        "fixed_params": {
            "ExtraTreeClassifier": {
                "min_samples_leaf": 4,
                "max_depth": 10,
                "class_weight": "balanced"
            }
        }
    }
},
```

Via the configuration, we are instructing the classifier to test two different classification models: **Random Forest** and **ExtraTreeClassifier**.

For forecast, you can load the [AutoMLResult](/api/utils.automl/#automlresults) object saved, and load the model from it:
```python
@pipeline
def forecast(self, data):
    # load the model
    model = self.get_object('model')

    # predict
    predictions = model.predict(data)

    # store the predictions
    self.add_predictions(data.index, predictions, [], prediction_type='forecast')
``` 
At this point, your **crash_course_data_manager.py** file should look like this:

```python
from mlapp.managers import DataManager, pipeline
import pandas as pd
import numpy as np
from mlapp.utils.features.pandas import extend_dataframe


class CrashCourseDataManager(DataManager):
    # -------------------------------------- train pipeline -------------------------------------------
    @pipeline
    def load_train_data(self, *args):
        data = pd.read_csv(self.data_settings["file_path"])
        return data

    @pipeline
    def clean_train_data(self, data):
        # Drop samples with missing label
        data.dropna(inplace=True, subset=[self.model_settings['target']])

        # Extract the list of features excluding the variable to predict
        features = list(data.columns)
        features.remove(self.model_settings['target'])

        # Calculate the mean value for each feature
        default_values_for_missing = data[features].mean(axis=0).to_dict()

        # Fill any missing values using the previous calculation
        data = data.fillna(default_values_for_missing)

        # Store the calculated missing values
        self.save_metadata('missing_values', default_values_for_missing)
        return data

    @pipeline
    def transform_train_data(self, data):
        data = self._transform_data(data)
        self.save_dataframe('features', data)
        return data

    @pipeline
    def load_forecast_data(self, *args):
        data = pd.read_csv(self.data_settings["file_path"])
        return data

    @pipeline
    def clean_forecast_data(self, data):
        # get the missing values
        default_values_for_missing = self.get_metadata('missing_values', default_value={})

        # fill the missing values
        data = data.fillna(default_values_for_missing)
        return data

    @pipeline
    def transform_forecast_data(self, data):
        # apply transformations
        data = self._transform_data(data)
        return data

    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()

    def _transform_data(self, data):
        # extend data frame
        conf_extend_dataframe = self.data_settings.get('conf_extend_dataframe', {})
        data = extend_dataframe(
            data,
            y_name_col=self.model_settings.get('target'),
            lead_order=conf_extend_dataframe.get('lead', 0),
            lag_order=conf_extend_dataframe.get('lag', 0),
            power_order=conf_extend_dataframe.get('power', 0),
            log=conf_extend_dataframe.get('log', False),
            exp=conf_extend_dataframe.get('exp', False),
            sqrt=conf_extend_dataframe.get('sqrt', False),
            poly_degree=conf_extend_dataframe.get('interactions', 0),
            inverse=conf_extend_dataframe.get('inverse', False)
        )
        return data

```

And your file **crash_course_model_manager.py** file should look like this:

```python
from mlapp.managers import ModelManager, pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from mlapp.utils.automl import AutoMLPandas


class CrashCourseModelManager(ModelManager):
    @pipeline
    def train_model(self, data):
        # splitting X and y
        target = self.model_settings['target']
        y = data[target]
        X = data.drop(target, axis=1)

        # split the data to test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.model_settings.get('train_percent'))

        # run auto ml
        result = AutoMLPandas('multi_class', **self.model_settings.get('auto_ml', {})) \
            .run(X_train, y_train, X_test, y_test)

        # print report
        result.print_report(full=False)

        # save results
        self.save_automl_result(result)

    @pipeline
    def forecast(self, data):
        # load the model
        model = self.get_object('model')

        # predict
        predictions = model.predict(data)

        # store the predictions
        self.add_predictions(data.index, predictions, [], prediction_type='forecast')
    
    @pipeline
    def refit(self, data):
        raise NotImplementedError()

```

Finally, your **crash_course_train_config.json** file should look like this:

```json
{
  "pipelines_configs": [
      {
      "data_settings": {
          "file_path": "data/glass.csv",
          "conf_extend_dataframe": {
              "log": true,
              "exp": true,
              "interactions": 2
          }
      },
      "model_settings": {
          "train_percent": 0.8,
          "target": "Type",
          "auto_ml": {
              "models": ["RandomForest", "ExtraTreeClassifier"],
              "hyper_params": {
                  "RandomForest": {
                      "max_depth": "range(1, 3)",
                      "n_estimators": [400, 700, 1000]
                  },
                  "ExtraTreeClassifier": {
                      "n_estimators": [10, 50, 200],
                      "min_samples_split": [2, 10]
                  }
              },
              "fixed_params": {
                  "ExtraTreeClassifier": {
                      "min_samples_leaf": 4,
                      "max_depth": 10,
                      "class_weight": "balanced"
                  }
              }
          }
      },
        "job_settings": {
          "asset_name": "crash_course",
          "pipeline": "train"
        }
      }
  ]
}
```

## Run the Improved Asset

Execute the `run.py` with the train pipeline check out the new result you got!

Now that you have a more mature classifier, you can play around with different train configurations. Here are a few suggestions below:

## Configuration Exploration

It is now possible to explore different configuration based on some data analysis to get the best possible model.

Below are a few examples we prepared.

**Configuration with more feature transformations:**
```json
{
    "pipelines_configs": [
        {
            "data_settings": {
                "file_path": "data/glass.csv",
                "conf_extend_dataframe": {
                    "log": true,
                    "exp": true,
                    "sqrt": true,
                    "interactions": 2,
                    "inverse": true
                }
            },
            "model_settings": {
                "train_percent": 0.8,
                "target": "Type",
                "auto_ml": {
                    "models": ["RandomForest", "ExtraTreeClassifier"],
                    "hyper_params": {
                        "RandomForest": {
                            "max_depth": "range(1, 3)",
                            "n_estimators": [400, 700, 1000]
                        },
                        "ExtraTreeClassifier": {
                            "n_estimators": [10, 50, 200],
                            "min_samples_split": [2, 10]
                        }
                    },
                    "fixed_params": {
                        "ExtraTreeClassifier": {
                            "min_samples_leaf": 4,
                            "max_depth": 10,
                            "class_weight": "balanced"
                        }
                    }
                }
            },
            "job_settings": {
                "asset_name": "crash_course",
                "pipeline": "train"
            }
        }
    ]
}
```

**Configuration with more modelling options:**
```json
{
    "pipelines_configs": [
      {
    "data_settings": {
        "file_path": "data/glass.csv",
        "conf_extend_dataframe": {
            "log": true,
            "exp": true,
            "sqrt": true,
            "interactions": 2,
            "inverse": true
        }
    },
    "model_settings": {
        "train_percent": 0.8,
        "target": "Type",
        "auto_ml": {
            "models": ["ExtraTreeClassifier", "RandomForest", "KnearestNeighbors"],
            "hyper_params": {
                "RandomForest": {
                    "max_depth": [5, 7, 10],
                    "n_estimators": [400, 700, 1000]
                },
                "ExtraTreeClassifier": {
                    "n_estimators": [10, 50, 200],
                    "min_samples_split": [2, 10]
                },
                "KnearestNeighbors": {
                    "n_neighbors": [1, 2, 3, 5],
                    "p": [1, 2, 3],
                    "leaf_size": [30, 60],
                    "algorithm": ["ball_tree", "kd_tree", "brute"]
                }
            },
            "fixed_params": {
                "ExtraTreeClassifier": {
                    "min_samples_leaf": 4,
                    "max_depth": 10,
                    "class_weight": "balanced"
                }
            }
        }
    },
    "job_settings": {
        "asset_name": "crash_course",
        "pipeline": "train"
    }
  }]
}
```