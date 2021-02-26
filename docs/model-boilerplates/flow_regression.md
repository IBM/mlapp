# Flow Regression

This boilerplate should not be used for a realistic scenario. It's an example of how you can combine multiple assets with a flow for demonstration purposes.
 
> Note: an elaborated explanation on flow can be found [here](/concepts/flow).

This asset consists an example of a flow config with two models of type [Advanced Regression](/model-boilerplates/advanced_regression), and a flow summarizing model named **flow_regression**.

The first model in the pipeline is doing a forecast. It accepts nothing as input from predecessor, 
but returns the `features_df` and the predictions to it. The second model in the pipeline also does a forecast, 
it accepts the `feature_df` and the prediction from previous stage and returns the prediction. 
The [flow manager](/api/managers.flow_manager) will forward all the outputs from all stages to the **flow_regression** asset. 
The later will sum all predictions by the index and will store the `feature_df`.

```python
flow_config = {
    "pipelines_configs":
        [
            # first pipeline's config
            {   
                "data_settings": {
                    "index_columns": ["index"]
                },
                "model_settings": {
                },
                "flow_settings":
                    {
                        "input_from_predecessor": [],
                        "return_value": ["df", "predictions"]

                    },
                "job_settings": {
                    "model_name": "advanced_regression",
                    "model_id": "97a765bb-b993-41f0-8f22-bfafb857b05e",
                    "pipeline": "forecast"
                }
            },
            # second pipeline's config
            {   
                "data_settings": {
                    "index_columns": ["index"]
                },
                "model_settings": {
                },
                "flow_settings":
                {
                    "input_from_predecessor": ["df", "predictions"],
                    "return_value": ["predictions"]
                },
                "job_settings": {
                    "model_name": "advanced_regression",
                    "model_id": "97a765bb-b993-41f0-8f22-bfafb857b05e",
                    "pipeline": "forecast"
                }
            },
        ],
    # flow summary config
    "flow_config": {   
        "data_settings": {
            # load data
            "flow_return_features": ["df"],
            "flow_return_data": "predictions",
            "data_index": ["index"],

            # features handling configurations
            "data_handling": {
                "agg_on_columns": ["y_hat"],
                "agg_function_dataframe": "np.sum"
            }
        },
        "job_settings": {
            "model_name": "flow_regression",
            "pipeline": "forecast_flow"
        }
    }
}
```


 