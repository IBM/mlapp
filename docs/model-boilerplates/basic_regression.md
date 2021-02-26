# Basic Regression

This asset is implementing a basic regression model handling continuous predicted variables.


Good examples for usage in this boilerplate are house price predictions, 
product sales, driving time, and any other quantity/continuous value. 
However, its main purpose is to be a simple introductory for the ML App framework concepts.

The boilerplate is using diabetes data set to predict the fasting 
blood sugar level of pre-diagnosed diabetes patients.
Features are blood measures and other personal characteristics.

## 1. Basic Regression Data Manager
The **_BasicRegressionDataManager_** class demonstrates how to **load** the data 
from local path and **clean** it using configurable instructions 
(e.g, which percent of NULLs will cause removal of a feature). 
The cleaning also demonstrate that one should save features' mean 
in order to fill in missing values in forecast tasks - by calling the parent class function `self.save_missing_values(missing_values)`.

**Transform** the data is plain and shows how to drop selected features 
that were indicated for drop in the config file .

This class also demonstrates how one can save time by calling the same 
inner functionalities both in train and forecast data handling tasks.

## 2. Basic Regression Model Manager

The **_BasicRegressionModelManager_** class builds a model and also implements 
the prediction task given a trained model.
**train_model** shows how one can use pre built functionality of our framework 
to save development time, for example, splitting the data to X and y, 
to train and test, use auto_ml functionality that selects the best model, 
the best hyper params  and best feature selection. 
It also shows how one can specify the different options for models 
(here one can specify whether which models to use under the "linear_models" auto_ml functionality.
 Here we selected to use Ridge and Lasso and no hyper param search.
 See more option in the ML App utilities documentation under [API > sklearn > model_utilities](/api/sklearn.model_utilities/).
 
Once the best model is selected, the data scientist is shown how to store it and 
store any relevant metadata on the selected model, then use those outputs in the **forecast** tasks.

> Note: the data scientist can create different images out-of-the-box by specifying the 
visualization methods of the framework, in the config file. See [advanced_regression](/model-boilerplates/advanced_regression) boilerplate for an example.

Once a model is trained, all its outputs is given a unique run id. 

## 3. Configs Directory
It is worth going through the different config files to see the difference tasks can be done using the asset.

Each config in this boilerplate gives an example of using a pre-built Pipeline. 
Please refer to [Concepts > Pipelines](/concepts/pipelines) for further information on pre built Pipelines.

#### 3.1. basic_regression_train_config
 
Runs a **train** pipeline that is described above. It runs through the following functions :
 `['load_train_data', 'clean_train_data', 'transform_train_data', 'train_model']`

 - This config describes in the **_data_settings_**  section what data to load and from where,
 how to clean the data and which features to keep in the **data_handling part**
 - This config describes in the **_model_settings_** how to split train/test, 
 and which algorithms to run in the auto_ml functionality
 
#### 3.2. basic_regression_forecast_config

Runs a **forecast** pipeline, that runs the following functions :
`['load_forecast_data', 'clean_forecast_data', 'transform_forecast_data', 'forecast']`,
 - This config describes in the **_data_settings_**  section what data to load for predictions and from where,
 how to clean the data and prepare it for forecast the  **data_handling part**
- This config describes in the **_job_settings_**  section to take
 the **latest** model that was run by the data scientist and not necessarily the best one.
 To select the best model, one need to pass the run_id of that model.
 

#### 3.3. basic_regression_feature_engineering_config

Runs a features' data set creation pipeline that automatically also saves the features using the parent class pipeline function **cach_features()**:
`['load_train_data', 'clean_train_data', 'transform_train_data', 'cache_features']`,
 - This config describes in the **_data_settings_**  section what data to load and from where,
 how to clean the data and which features to keep in the **data_handling part**

#### 3.4. basic_regression_reuse_features_and_train_config

Runs a pipeline that loads the existing features set using the parent class pipeline function **load_features**
and running them through the **train_model** function
['load_features','train_model'],
 - The config is expected to specify what is the run id 
 of the run creating the features' dataset. if "latest" is used then the latest data id existing in the path is used  

#### 3.5. basic_regression_custom_pipeline_config

Here we specify in the "job_settings" section of the config what pipeline functions we want to run and by which order.
In this pipeline example we will run through load_data, clean data and then save the clean dataset for next reuse.

```json
{
  "job_settings": {
    "asset_name": "basic_regression",
    "pipeline": ["load_train_data", "clean_train_data", "cache_features"]
  }
}
```

#### 3.6. basic_regression_train_step_config

This config shows how to use the basic regression as part of a Flow.
More specific, it adds a section to the config named "flow_settings":

```json
{
  "flow_settings":{
    "input_from_predecessor":["contributer_name"],
    "return_value":["predictions"]
  }
}
```
            
- For more information on Flows, please check out [Concepts > Flow](/concepts/flow) and also see the example at the [Advanced Course](/advanced-topics/introduction).  

