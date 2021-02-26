# Multi-Step Pipeline Example

## 1. Prequisites

- Steps 1-3 at [End-to-End Deployment](/integrations/azureml/getting-started).
- AzureML Workspace.
- AzureML Compute Target available.
- MLApp's Crash Course.

> Note: You can install the Crash Course to your MLApp project with the MLApp CLI **boilerplates** command in the root of your project:
```bash
mlapp boilerplates install crash_course
```

- We'll be using MLApp's Flow. For a better understanding of MLApp's Flow you can check [here](/concepts/flow), and for an example you check the [Advanced Topics](/advanced-topics/part-2-flow).

## 2. Introduction

We'll create a Flow containing two pipelines: 
- `feature_engineering` - will be responsible on the feature engineering part of the Crash Course and will output the created features.
- `["load_features_from_predecessor", "train_model"]` - will load the features from the previous Pipeline and will do the modelling part of the Crash Course.

In AzureML we'll be able to run each Pipeline in a separate step using a different Compute Target.

## 3. Create a Flow in MLApp

1. Create a new file called `crash_course_train_flow_config.json` at `assets > crash_course > configs`. Pass in the following to its contents:

```json
{
  "pipelines_configs": [
    {
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": "feature_engineering"
      },
      "flow_settings": {
          "input_from_predecessor": [],
          "return_value": ["features"]
      }
    },
    {
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": ["load_features_from_predecessor", "train_model"]
      },
      "flow_settings": {
          "input_from_predecessor": ["features"],
          "return_value": []
      }
    }
  ]
}
```

> Note: we have added the `flow_settings` key in both Pipelines. One to pass the features and the other to receive.

2. Set up the `data_settings` for the first the `feature_engineering` Pipeline. Copy it from the `data_settings` of the configuration here: `assets > crash_course > configs > crash_course_train_config.json`.

3. Add the following `model_settings` to the first Pipeline:

```json
{ 
  ...
  "model_settings": {
    "target": "Type"
  },
  ...
}
```

> Note: this is required as the Crash Course's Data Manager relies on this configuration to know which is the target column.

4. Set up the `model_settings` for the second the `reuse_features_and_train` Pipeline. Copy it from the `model_settings` of the configuration here: `assets > crash_course > configs > crash_course_train_config.json`.

5. Your final configuration should look like this:

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
        "target": "Type"
      },
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": "feature_engineering"
      },
      "flow_settings": {
        "input_from_predecessor": [],
        "return_value": [
          "features"
        ]
      }
    },
    {
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
      "flow_settings": {
          "input_from_predecessor": ["features"],
          "return_value": []
      },
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": ["load_features_from_predecessor", "train_model"]
      }
    }
  ]
}
```

## 4. Add the `load_features_from_predecessor` Pipeline Function

Add the following Pipeline Function in the Crash Course's Data Manager:

```python
@pipeline
def load_features_from_predecessor(self, *args):
    data = None
    # input from predecessor pipeline
    if 'features' in self.flow_settings.get('input_from_predecessor', []):
        predecessor_input = self.get_input_from_predecessor_job('features')
        if predecessor_input is not None and len(predecessor_input):
            print('Found input from predecessor Pipeline!')
            data = predecessor_input[0]

    if data is None:
        raise Exception("No data was found from predecessor")

    return data
```

> Note: for more information on **Pipeline Function** check the example at Part 3 of the Advanced Course: [Custom Pipeline](/concepts/pipelines/#4-adding-a-custom-pipeline). 

## 5. Run the Flow locally

Open the `run.py` file and run the configuration created in part 3:

```json
{
    "asset_name": "crash_course",
    "config_path": "assets/crash_course/configs/crash_course_train_flow_config.json",
    "config_name": "crash_course_config"
}
```

## 6. Create an AzureML Pipeline Endpoint

1. Use the following command to create your AzureML Pipeline:

```
mlapp aml publish-multisteps-pipeline <pipeline_endpoint_name>
```

You will be prompt for how many steps. The number of steps should be the same as the number of pipelines in your flow.

In our case it will be **2**.

For each step you will choose the AzureML Compute Target attached to the step. Use the Compute Target available in your AzureML Workspace.

## 7. Use Pipeline Endpoint in the AzureML UI: https://ml.azure.com/

1. Go to "Endpoints" in the left navbar -> select "Pipeline endpoints" tab.
2. Go to <pipeline_endpoint_name>
3. Click on “Submit”.
4. Create an experiment name.
5. Split your Flow Config to the different steps into 2 configurations each one containing only one Pipeline:

First configuration:
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
        "target": "Type"
      },
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": "feature_engineering"
      },
      "flow_settings": {
          "input_from_predecessor": [],
          "return_value": ["features"]
      }
    }
  ]
}
```

Second configuration
```json
{
  "pipelines_configs": [
    {
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
      "flow_settings": {
          "input_from_predecessor": ["features"],
          "return_value": []
      },
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": ["load_features_from_predecessor", "train_model"]
      }
    }
  ]
}
```

6. Place the created configurations in the right order: first Pipeline to `config0` and second Pipeline to `config1`.
7. Hit the "Submit" button.
8. Go to “Experiments” on the left navbar. Select your experiment.
9. See results  - you will see your run in there as “running”.