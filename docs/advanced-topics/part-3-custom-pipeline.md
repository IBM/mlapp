# Part 3 - Custom Pipeline

## 1. Prerequisites - Crash Course and Advanced Course Assets

Make sure you have finished [Part 1](/advanced-topics/part-1-boilerplates) and [Part 2](/advanced-topics/part-2-flow) of the Advanced Course in your current project.

Run the flow configuration created in part 2 via `run.py` and see it has no errors:
```json
{
    "config_path": "configs/part2_flow_config.yaml"
}
```

## 2. Custom Pipeline - Creating your own customized Pipeline

The Advanced Course is using the **Classification Boilerplate** which is more advanced in it's clean & transformation of the data.

There is no need to run the load-clean-transform feature engineering process in both Crash Course and Advanced Course assets.

Lets create a new custom pipeline for the Advanced Course asset that loads the ready-features directly and goes straight into the `train_model` pipeline function of the Model Manager.


## 3. Update the Assets

Unlike the Flow we created in Part 2, where we passed only enriched features, we'll want to pass the whole data frame containing the features.

In `assets > advanced_course > advanced_course_data_manager.py`:

```python
@pipeline
def transform_train_data(self, data):
    results_df = self._transform_data(data)
    print("------------------------- return final features to train -------------------------")

    # pass the feature engineered data to the next Pipeline 
    if 'features' in self.flow_settings.get('return_value', []):
        self.save_dataframe('features', results_df)

    return results_df
```

In `assets > crash_course > crash_course_data_manager.py` create a new Pipeline function called `load_features_from_predecessor`:

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

!!! note "@pipeline"
    
    when creating a new Pipeline function you must use the `@pipeline` decorator so the MLapp framework can recognize it as such.
    
    You can decide whichever arguments you wish to pass into your custom Pipeline function. If empty keep `*args`.


## 4. Update the Flow Configuration

Copy the contents of `configs > part2_flow_config.yaml` into a new configuration `configs > part3_flow_config.yaml`.

At `configs > part3_flow_config.yaml` update the `flow_settings` key name according to the changes we made in the assets from `enriched_features` to `features`:

```json
{
    "pipelines_configs": 
    [
      { 
        "data_settings": { ... },
        "flow_settings": {
            "input_from_predecessor": [],
            "return_value": ["features"]
        },
        "job_settings": { 
            "asset_name": "advanced_course",
            "pipeline": "feature_engineering"
         }
      },
      {
        "model_settings": { ... },
        "flow_settings": {
            "input_from_predecessor": ["features"],
            "return_value": []
        }, 
        "job_settings": { 
          "asset_name": "crash_course",
          "pipeline": "train" 
        } 
      }
    ]
}
```

## 5. Use Custom Pipeline

We want the Pipeline of the Crash Course to run the new Pipeline function we created: `load_features_from_predecessor` and afterwards the  `train_model` Pipeline function.

There are two options to do this:

**Option 1:** In the config at `job_settings` at they key `pipeline` pass in a list of functions we plan to run, e.g.
`"job_settings": { ... , "pipline": ["load_features_from_predecessor", "train_model"]}`

**Option 2:** In the `config.py` file of your root project you can add a key `pipelines` and define customized pipelines and use them in your configurations. E.g. define `"pipelines": { "train_using_predecessor_features": ["load_features_from_predecessor", "train_model"]}` in your `config.py` file and in your config use: `"job_settings": { ... , "pipline": "train_using_predecessor_features"}`

Lets use **Option 1** and update `configs > part3_flow_config.yaml`:
```json
{
    "pipelines_configs": 
    [
      { 
        "data_settings": { ... },
        "flow_settings": {
            "input_from_predecessor": [],
            "return_value": ["features"]
        },
        "job_settings": { 
            "asset_name": "advanced_course",
            "pipeline": "feature_engineering"
         }
      },
      {
        "model_settings": { ... },
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

!!! note "data_settings"

    As we're doing all the data processing in the first Pipeline, we can remove safetly the `data_settings` from the second Pipeline config.


Go ahead and run this new configuration via `run.py`:

```json
{
    "config_path": "configs/part3_flow_config.yaml"
}
```

> Note: you should be able to see the message `Found input from predecessor Pipeline!` in the logs which means data was passed between the Pipelines!

That's it! You have used your first custom Pipeline!


