# Part 5 - Forecast Pipeline

Now that the train pipeline has been successfully tested, it is time to develop the forecast pipeline and test it on some samples.

We have prepared a forecast dataset at **data/glass_forecast.csv**, which contains unlabeled samples of the glass dataset.

First implement the _load_forecast_data_, _clean_forecast_data_ and _transform_forecast_data_ functions in the **assets/crash_course/crash_course_data_manager.py**:

```python
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
    # apply same transformations for forecast
    features_to_log = self.data_settings.get('features_to_log', [])
    for feature in features_to_log:
        data[feature] = data[feature].apply(np.log)
    return data
```
Here we are loading unlabelled samples and replacing the missing values with the average feature values we saved during the train pipeline.
Then, we apply the same transformations done during the training pipeline.

Next, implement the _forecast_ function in **assets/crash_course/crash_course_model_manager.py**:

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

We are using a method of the model_manager called `add_predictions`. This method formats the predictions and adds saves them as a data frame for storage purpose.

Finally, modify the forecast base configuration **assets/crash_course/configs/crash_course_forecast_config.json**. Update the `model_id` property in the `job_settings` with the `run_id` from the train pipeline ran in the part 4 (or use `latest`).

```json
{
  "pipelines_configs": [
    {
      "data_settings": {
        "file_path": "data/glass_forecast.csv"
      },
      "model_settings": {

      },
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": "forecast",
        "model_id": "<run_id>"
      }
    }
  ]
}
```

!!! note "Don't forget to update the _run_id_"
    For the forecast pipeline to run, MLApp needs to know which run to load and use. Leaving this key blank will prevent the pipeline from working properly.
    
!!! tip "Using _latest_ instead of _run_id_"
    If you just want to use the latest run you can put in `latest` instead of the `run_id`.

!!! tip "Forecast's _data_settings_ configuration"
    
    During the forecast pipeline, the `data_settings` configuration is loaded from the training pipeline run. Therefore, the same `data_settings` that was applied in the training will be applied in the forecast. 
    
    You can override any settings by explicitly defining them (like the `file_path`).

Just as before, execute the forecast pipeline via the `run.py` file. Add the following config object to the configs list in **run.py**. Make sure to comment out the train pipeline.

```json
{
    "asset_name": "crash_course",
    "config_path": "assets/crash_course/configs/crash_course_forecast_config.json"
}
```

A new file will appear in the output folder **output/<run\_id\>_crash_course.predictions.csv**. 

!!! note "Predictions CSV File"

    * **index**: Id of the sample
    
    * **y_true**: y_true if it exists (in a forecast settings, it doesn't)
    
    * **y_hat**: y_hat displays the estimation value
    
    * **type**: data type: train=1, test=2, forecast=3.

**Congratulations! You have successfully built a simple classifier in the MLApp framework, including implementing a fully functional train and forecast pipeline.**

<br/>
We can move on to [improving the asset using MLApp's utilities](/crash-course/part-6-improving-asset).
