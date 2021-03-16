# Part 2 - Developing the Data Manager

The Data Manager is in charge of the data pipeline: loading the data, cleaning the data and transforming the data.

First, open the file **assets/crash_course/crash_course_data_manager.py**. Add a new line at the top of the file to import the pandas and numpy libraries:

```python
import pandas as pd
import numpy as np
```

Now you are ready to develop the train data pipeline by filling in the logic for the _load_train_data_, _clean_train_data_ and _transform_train_data_ functions (we will focus on the train pipeline for now - later we will address the forecast pipeline).

To load the CSV into memory, we will simply use the **pandas** `read_csv` function:

```python
@pipeline
def load_train_data(self, *args):
    data = pd.read_csv(self.data_settings["file_path"])
    return data
```

Notice the first example of configuration injection - instead of hard coding the file path, we are injecting it into the code from the data_settings dictionary. This will allow a future data scientist to change the file path dynamically, making the asset much more reusable.

!!! note "Always return your data!"

     Notice that all functions in the data manager return a data frame. This is important, as this data frame is injected as a parameter into the next function in the pipeline - i.e., load, then clean, and finally transform. Forgetting to return data at the end of each function will result in an error.


In order to support this configuration option, open the file **assets/crash_course/configs/crash_course_train_config.json** and add the option to data_settings:

```json
"data_settings": {
    "file_path": "data/glass.csv"
}
```


Next, add some functionality for cleaning up the data. In this case, we will drop any samples that have a missing tag.

```python
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
```

Notice here that the model target is also an injected configuration option.

```json
"model_settings": {
    "target": "Type"
}
```

!!! note "We are doing a few things in the _clean_train_data_ function"

    * First, we drop any sample with missing label.
    
    * Then we calculate the means of each feature (excluding the target).
    
    * Then, fillna is used to replace the missing values by the average values.
    
    * These average values must be stored for later, as they will be needed in the forecast pipeline to fill in missing values in the same way. The **self.save_missing_values** method is used to store the calculations of feature averages from the train data.

!!! tip "Using Managers' _self_ object"

    The `self` here is of type `DataManager` and can be used to store any valuable information generated during the training data processing, that might be useful for review by data scientists or required in the forecast pipeline. You can use the manager to store:
    
    **Metadata**: store any important information that is JSON compatible such as integers, strings, floats, dictionaries or lists.
    
    **Run-time objects**: models, one-hot encoder, or anything that is not JSON compatible.
    
    **Data frames**: any dataframe you wish to save and reload later.
    
    **Image files**: store images or figures. For example, analysis generated using matplotlib.
    
    > For the full API Reference go [here](/api/managers/#shared-functions).
    


Finally, we will add some logic to transform the data to potentially increase predictive value. For example, we can decide to apply log to some features.

```python
@pipeline
def transform_train_data(self, data):
    features_to_log = self.data_settings.get('features_to_log', [])
    for feature in features_to_log:
        data[feature] = data[feature].apply(np.log)
    return data
```

```json
"data_settings": {
    "file_path": "data/glass.csv",
    "features_to_log": ["Na"]
}
```

At this point in the crash course, your file **assets/crash_course/crash_course_data_manager.py** file should look like this:

```python
from mlapp.managers import DataManager, pipeline
import pandas as pd
import numpy as np


class CrashCourseDataManager(DataManager):
    def __init__(self, config, *args, **kwargs):
        DataManager.__init__(self, config, *args, **kwargs)

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
        features_to_log = self.data_settings.get('features_to_log', [])
        for feature in features_to_log:
            data[feature] = data[feature].apply(np.log)
        return data

    # ------------------------------------ forecast pipeline -------------------------------------------
    @pipeline
    def load_forecast_data(self, *args):
        raise NotImplementedError("should return data")
    
    @pipeline
    def clean_forecast_data(self, data):
        return data

    @pipeline
    def transform_forecast_data(self, data):
        return data
    
    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()

```

And your file **assets/crash_course/configs/crash_course_train_config.json** should look like this:

```json
{
  "pipelines_configs": [
    {
      "data_settings": {
        "file_path": "data/glass.csv",
        "features_to_log": ["Na"]
      },
      "model_settings": {
          "target": "Type"
      },
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": "train"
      }
    }
  ]
}
```

<br/>
Congrats, the data processing part of the pipeline is complete! We can move on to [developing the model manager](/crash-course/part-3-developing-the-model-manager).