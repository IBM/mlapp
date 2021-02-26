# Flow

## What is a Flow?
A Flow is a run of several pipelines one after the other. 
It is needed when the data science requires input from previous model in order to operate on a second model. 
A common example is when the predecessor model's output is necessary as a feature/target in the successor model. 
For example, one may want to cluster his samples and put that as an input for the second model, which was built for each cluster separately.
In such cases, the Data scientist will train 2 models- clustering and then some regression per cluster. 
When forecasting, we will need the flow to start by cluster prediction and moving this information to the next model which will choose the adequate model to activate.
Finally we may also want to be able to take some action on all stages outputs to run some post-Flow analysis. 
For example, if we break down the cost prediction model to minor models which compose the total cost (e.g. predict the cost of manufacturing, cost of shipment, cost of service per item), 
we may finally want to predict the total cost of the Flow by summing up the outputs from each model and return that as the prediction.

## What can be a Flow?

**One or more "regular" models in a "pipelines_configs"**.
 
These models are the data science models of the flow. e.g. cost_manufacturing, cost_shipment, cost_service models. 
These are built using the `mlapp asset create` functionality, and have a DataManager, a ModelManager and a config. 
The main difference is that these models need to be able to accept data from their predecessors if needed, for example, 
one may want to pass the predicted cost of manufacturing to the cost_service model, as those may be interrelated, therefore, 
the output of model No.1 should be passed to model No.2 in order for it to pass it to model No.3 in the flow. 
We can see an example below.

## Flow Features

#### Flow Input/Output

The way the predecessor information is passed to the successor pipeline is through the [DataManager](/api/managers.data_manager) and [ModelManager](/api/managers.model_manager) save functions of each asset such as [self.save_metadata_value](/api/managers.data_manager/#save_metadata_value) or [self.save_object](/api/managers.data_manager/#save_object). Then, load the information using the [self.get_input_from_predecessor_job](/api/managers.data_manager/#get_input_from_predecessor_job) function available in the Data Manager and Model Manager.
One must also make sure s/he returns the right information to be passed. 

The simplest output information can be **predictions** which is the key of the predictions for each model (the data scientist is expected to call the `ModelManager.add_prediction(self, primary_keys_columns, y_hat, y=pd.Series(), prediction_type='TRAIN')` to save the model's predictions). 
More complex information to pass can come from the DataManager i.e. some feature that was calculated. In that case, the data scientist is expected to store that feature by calling `DataManager.save_features(self, features_df)`.

#### Flow Summarising Model 
In case there needs to be post processing activity on all pipelines outputs, a flow summarizer can be added. 
The flow summarizer is an asset of its own, with a DataManager, a ModelManager and a config. 
It's input is the outputs from the entire flow models (but does not have to be exclusively that). 
The pipelines' input is accessible through the `*args` of the `DataManager.load_train_data(self, *args)` or `DataManager.load_forecast_data(self, *args)` in the DataManager instance (or whichever [Pipeline Function](/concepts/pipelines/#2-pipeline-functions) that is the entry function of the running Pipeline). 
For example, one may want to sum all `predictions` from all the models in the Flow. 
Then, the `DataManager.load_forecast_data()` logic will consume the data from `*args`, then in `transform_forecast_data()` one can concatenate all the predictions data frames, and sum the predictions on some given index. 
In the Model Manager, one should add the final sum to the `ModelManager.add_predictions()`. 
See an implementation in the [flow_regression](/model-boilerplates/flow_regression) example in the MLApp Boilerplates. 

> Note: the config for this model will use one of the following two out-of-the-box pipelines (as needs to be indicated in the `job_settings` section): **forecast_flow** or **train_flow**.

#### Flow Summary Config 

The general structure of the flow summarizer config is as follows:

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

**pipelines_configs**: list of configs for the flow sub-models, each config contains `flow_settings` as well.

**flow_config**: config of the flow-summarizing model.

>Note: One can have only one pipeline run and a flow summarizer for post processing, as the flow. 


## Flow Example

See the [flow_regression](/model-boilerplates/flow_regression) boilerplate.