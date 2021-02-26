# Pipelines

## 1. What is a Pipeline

A Pipeline is a set of stages/logic you wish to run in a specific order. 

A Pipeline contains one or more stages where each stage is represented by a **Pipeline Function**. 

## 2. Pipeline Functions

A Pipeline function is any function in your Data Manager or Model Manager that you want as part of your pipeline.

A pipeline is mandatory when running your asset in MLapp. When preparing a configuration use the `job_settings` and in it add a `pipeline` key with the value being the pipeline you wish to run:
```json
{
  "job_settings": {
    "asset_name": "crash_course",
    "pipeline": "train"
  }
}
```

> Note: the above configuration will be running the **train** pipeline with the **Crash Course** asset. 
 
## 3. Available Best-Practices Pipelines

MLApp has best-practice pipelines available for use: 

```json
{
    "train": ["load_train_data", "clean_train_data", "transform_train_data", "train_model"],
    "explore_data": ["load_train_data", "clean_train_data", "transform_train_data", "visualization"],
    "feature_engineering": ["load_train_data", "clean_train_data", "transform_train_data", "cache_features"],
    "reuse_features_and_train":["load_features","train_model"],
    "forecast": ["load_forecast_data", "clean_forecast_data", "transform_forecast_data", "forecast"],
    "predictions_accuracy": ["load_target_data", "load_predictions", "evaluate_prediction_accuracy"],
    "retrain": ["load_train_data", "clean_train_data", "transform_train_data", "train_model"]
}
```

## 4. Adding a Custom Pipeline

You can also add your own customized pipelines to the framework.

#### 4.1. Adding Pipeline Function

When creating a new Pipeline function you must use the `@Pipeline` decorator so the MLapp framework can recognize it as such.

You can decide whichever arguments you wish to pass into your custom Pipeline function. If empty keep `*args`.

Example:
```python
@pipeline
def custom_function(self, *args):
    print("Hello! I'm running a custom Pipeline function!")
    return True
```

To use this Pipeline Function there are two ways: via configuration and via `config.py`.

#### 4.2. Via Configuration
 
In the config at `job_settings` at they key `pipeline` pass in a list of functions you plan to run, e.g.:
```json
{
  "job_settings": { 
    "pipline": ["custom_function", "train_model"]
  }
}
```

#### 4.3. Via `config.py` file in project

In the `config.py` file of your root project you can add a key `pipelines` and define customized pipelines and use them in your configurations. 

E.g. define in your `config.py` file:
```json
{
  "pipelines": { 
    "custom_pipeline": ["custom_function", "train_model"]
  }
}
```

And in your config use: 
```json
{
  "job_settings": { 
    "pipline": "custom_pipeline"
  }
}
```
