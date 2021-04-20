# Part 2 - Flow

## 1. Prerequisites - Crash Course and Advanced Course Assets

Make sure you have finished [Part 1](/advanced-topics/part-1-boilerplates) of the Advanced Course in your current project.

Run again the train Pipeline using the configuration file created in Part 1 `advanced_course_glass_train_config.yaml` via `run.py` and see it has no errors:
```json
{
    "asset_name": "advanced_course",
    "config_path": "assets/advanced_course/configs/advanced_course_glass_train_config.yaml",
    "config_name": "advanced_course_config"
}
```

Be sure that you have your [Crash Course](/crash-course/introduction) asset prepared as well. 

You can always install it again with the MLApp's CLI **boilerplates** command in the root of your project:
```bash
mlapp boilerplates install crash_course
```

Run a train Pipeline via `run.py` and see it has no errors:
```json
{
    "asset_name": "crash_course",
    "config_path": "assets/crash_course/configs/crash_course_train_config.yaml",
    "config_name": "crash_course_config"
}
```

## 2. Flow - Running Multiple Pipelines Consecutively 

Up until now you've been running configurations with one Pipeline - for example a usual configuration:

```json
{
    "pipelines_configs": 
    [
      {
          "data_settings": {
            ...
          },
          "model_settings": {
            ...
          },
          "job_settings": {
            ...
          }
      }
  ]
}
```

MLApp framework allows you to run a Flow which is a run of more than one Pipeline:

```json
{
    "pipelines_configs": 
    [
      // first pipeline
      {
          "data_settings": {
            ...
          },
          "model_settings": {
            ...
          },
          "job_settings": {
            ...
          }
      },
      // second pipeline
      {
          "data_settings": {
            ...
          },
          "model_settings": {
            ...
          },
          "job_settings": {
            ...
          }
      }
  ]
}
```

Each Pipeline in the Flow can run a different asset you've built with the framework.

You can also pass data between each Pipeline in a simple manner and in case you want a final summary Pipeline to run an action with all the outputs from the previous Pipelines you can do it in this way:

```json
{
    "pipelines_configs": 
    [
      // first pipeline
      {
          "data_settings": {
            ...
          },
          "model_settings": {
            ...
          },
          "flow_settings": {
            "input_from_predecessor": [...],
            "return_value": [...]
          },
          "job_settings": {
            ...
          }
      },
      // second pipeline
      {
          "data_settings": {
            ...
          },
          "model_settings": {
            ...
          },
          "flow_settings": {
            "input_from_predecessor": [...],
            "return_value": [...]
          },
          "job_settings": {
            ...
          }
      }
  ],
  // flow configuration - final summarizer pipeline
  "flow_config": {
      "data_settings": {
        ...
      },
      "model_settings": {
        ...
      },
      "job_settings": {
        ...
      }
  }
}
```

In the next steps we will see this in action...

## 3. Create a New Flow Configuration

Create a new directory `configs` in the root of your projects.

Create a new configuration file at `configs > part2_flow_config.yaml`.

Copy both the train configuration from the glass train configuration from the Advanced Course and the train configuration from the Crash Course and paste them one after the other in the array, in the following way:
```json
{
    "pipelines_configs": 
    [
      // advanced course config
      { "data_settings": { ... }, "model_settings": { ... }, "job_settings": { ... } },
      // crash course config
      { "data_settings": { ... }, "model_settings": { ... }, "job_settings": { ... } }
    ]
}
```

Result should be:

```json
{
  "pipelines_configs": [
    {
      "data_settings": {
        "local_data_csvs": [
          {
            "name": "glass",
            "path": "data/glass.csv"
          }
        ],
        "variable_to_predict": "Type",
        "data_handling": {
          "y_variable": {
            "type": "multi",
            "label_to_predict": ["1", "2", "3", "4", "5", "6", "7"]
          },
          "feature_remove_by_null_percentage": 0.3,
          "features_handling": {},
          "features_to_bin": [
            {"name": "RI", "bins": [1.516523, 1.519157]},
            {"name": "Na", "bins": [12.9075, 13.825]},
            {"name": "Mg", "bins": [2.115000, 3.600000]},
            {"name": "Al", "bins": [1.19, 1.63]},
            {"name": "Si", "bins": [72.28, 73.0875]},
            {"name": "K", "bins": [0.1225, 0.61]},
            {"name": "Ca", "bins": [8.24, 9.1725]},
            {"name": "Ba", "bins": [0.025]},
            {"name": "Fe", "bins": [0, 0.1]}
          ],
          "use_evaluator": false
        }
      },
      "model_settings": {
        "variable_to_predict": "Type",
        "down_sample_method": {"flag": false, "n_samples": 100, "seed": 1500 }, "train_percent": 0.8,
        "predict_proba_threshold": [0.05, 0.5, 0.95 ], "auto_ml": {
          "feature_selection": [
            {
              "method": "SelectKBest",
              "params": {
                "k": 7,
                "score_func": "chi2"
              }
            },
            {
              "method": "RFE",
              "params": {
                "estimator": "SVR(kernel='linear')",
                "n_features_to_select": 5,
                "step": 1
              }
            },
            {
              "method": "AllFeatures"
            }
          ],
          "estimator": "multi_class",
          "multi_class": {
            "models": ["MultiLogistic", "ExtraTreeClassifier", "KnearestNeighbors", "RandomForest"]
          }
        }
      },
      "job_settings": {
        "asset_name": "advanced_course",
        "pipeline": "train"
      }
    },
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
        "train_percent": 0.9,
        "target": "Type",
        "auto_ml": {
          "models": [
            "RandomForest",
            "ExtraTreeClassifier"
          ],
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

Go ahead and run this new configuration via `run.py`:

```json
{
    "config_path": "configs/part2_flow_config.yaml"
}
```

!!! note "part2_flow_config.yaml"

    Running this configuration will run a Flow with 2 Pipelines, first Pipeline would be the train of the Advanced Course asset (Classification Boilerplate), and the second pipeline the train of the Crash Course asset.

Congratulations! you have run a Flow. Let's explore one of the available features of the Flow.

## 4. Pass Data Between Pipelines

The Advanced Course is using the **Classification Boilerplate** which has more advanced data handling. 

One of it's feature engineering option is to bin features. We've seen good response in the classifier models to the binned `Mg` feature.

In the following example Flow, we will first run a `feature_engineering` Pipeline to create the enriched binned `Mg` feature and pass it on to our Crash Course `train` Pipeline.

We'll prepare the configuration `configs > part2_flow_config.yaml`:

### 4.1. Update the configuration

1.  We'll remove the `model_settings` key from the json as we're just using the `feature_engineering` Pipeline which only goes over the Pipeline functions of the Data Manager.

2.  We'll update the `pipeline` key in `job_settings` to `feature_engineering`.

3.  We'll add a key in the `data_settings` called `enriched_features` where we'll specify which features we want to select.

The configuration at `configs > part2_flow_config.yaml` should look like this:

```json
{
  "pipelines_configs": [
    {
      "data_settings": {
        "enriched_features": ["Mg"],
        "local_data_csvs": [
          {
            "name": "glass",
            "path": "data/glass.csv"
          }
        ],
        "variable_to_predict": "Type",
        "data_handling": {
          "y_variable": {
            "type": "multi",
            "label_to_predict": ["1", "2", "3", "4", "5", "6", "7"]
          },
          "feature_remove_by_null_percentage": 0.3,
          "features_handling": {},
          "features_to_bin": [
            {"name": "RI", "bins": [1.516523, 1.519157]},
            {"name": "Na", "bins": [12.9075, 13.825]},
            {"name": "Mg", "bins": [2.115000, 3.600000]},
            {"name": "Al", "bins": [1.19, 1.63]},
            {"name": "Si", "bins": [72.28, 73.0875]},
            {"name": "K", "bins": [0.1225, 0.61]},
            {"name": "Ca", "bins": [8.24, 9.1725]},
            {"name": "Ba", "bins": [0.025]},
            {"name": "Fe", "bins": [0, 0.1]}
          ],
          "use_evaluator": false
        }
      },
      "job_settings": {
        "asset_name": "advanced_course",
        "pipeline": "feature_engineering"
      }
    },
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
        "train_percent": 0.9,
        "target": "Type",
        "auto_ml": {
          "models": [
            "RandomForest",
            "ExtraTreeClassifier"
          ],
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

4. We'll add `flow_settings` to the Advanced Course:

```json
{
    "pipelines_configs": 
    [
      { 
        "data_settings": { ... }, 
        "model_settings": { ... }, 
        "flow_settings": {
            "input_from_predecessor": [],
            "return_value": ["enriched_features"]
        },
        "job_settings": { 
            "asset_name": "advanced_course",
            "pipeline": "feature_engineering"
         }
      },
      { 
        "data_settings": { ... },
        "model_settings": { ... },
        "job_settings": { 
          "asset_name": "crash_course",
          "pipeline": "train" 
        } 
      }
    ]
}
```

!!! note "input_from_predecessor"

    We are not expecting any input in the first Pipelines, hence `input_from_predecessor` is an empty list. We want to pass on the enriched feature so we'll add that to the `return_value` list.

5. We'll add `flow settings` to the Crash Course configuration:
```json
{
    "pipelines_configs": 
    [
      { 
        "data_settings": { ... },
        "model_settings": { ... },
        "flow_settings": {
            "input_from_predecessor": [],
            "return_value": ["enriched_features"]
        }, 
        "job_settings": { ... } },
      { 
        "data_settings": { ... }, 
        "model_settings": { ... }, 
        "flow_settings": {
            "input_from_predecessor": ["enriched_features"],
            "return_value": []
        }, 
        "job_settings": { ... }
       }
    ]
}
```

!!! note "return_value"

    We are not expecting to output any data in the second Pipeline, hence `return_value` is an empty list. We want to receive the enriched feature from the first Pipeline so we'll add that to the `input_from_predecessor` list.

### 4.2. Update the Assets' code

Once we set up the configuration file, we'll update the code in the Crash Course & Classification assets to support it.
1. Lets update the advanced course DataManager to save the data to be passed on:

In `assets > advanced_course > advanced_course_data_manager.py`:

```python
@pipeline
def transform_train_data(self, data):
    results_df = self._transform_data(data)
    # ------------------------- return final features to train -------------------------
    print("------------------------- return final features to train -------------------------")
    
    # pass the feature engineered data to the next Pipeline 
    if 'enriched_features' in self.flow_settings.get('return_value', []):
        self.save_dataframe('enriched_features', results_df[self.data_settings.get('enriched_features', [])])
    
    return results_df
```

> Note: Advanced Course model can be still used in Pipeline just by itself, hence we pass the `enriched_features` only in case we set it in the `flow_settings`.

2. In the crash_course DataManager use the input received from the predecessor Pipeline: 

In `assets > crash_course > crash_course_data_manager.py` update the `trasform_train_data()` Pipeline function:

```python
@pipeline
def transform_train_data(self, data):
    # apply transformations
    data = self._transform_data(data)

    if 'enriched_features' in self.flow_settings.get('input_from_predecessor', []):
        predecessor_input = self.get_input_from_predecessor_job('enriched_features')[0]
        data = pd.merge(
            data, predecessor_input, how='left', left_index=True, right_index=True, suffixes=("", "_binned"))

    # store dataframe:
    self.save_dataframe('features', data)

    return data
```

> Note: Crash Course asset can be still used in Pipeline just by itself, hence we merge the `enriched_features` data only in case we set it in the `flow_settings`.

Go ahead and run this new configuration via `run.py`:

```json
{
    "config_path": "configs/part2_flow_config.yaml"
}
```

That's it! You have run your first Flow with passing data between Pipelines! 

You can check and see if the model scores improved...


