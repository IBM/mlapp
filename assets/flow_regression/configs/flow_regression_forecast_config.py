flow_regression_forecast_config = {
    "pipelines_configs": [
        {
            "data_settings": {
                # load data
                "data_sources": {
                    # local file
                    "local": {
                        "file_paths": ["data/diabetes.csv"]  # local files
                    }
                },
            },
            "flow_settings": {
                "input_from_predecessor": [],
                "return_value": ["predictions"]

            },
            "job_settings": {
                "asset_name": "advanced_regression",
                "pipeline": "forecast",
                "model_id": "latest",
                "data_id": "latest"
            }
        },
        {
            "data_settings": {
                # load data
                "data_sources": {
                    # local file
                    "local": {
                        "file_paths": ["data/diabetes.csv"]  # local files
                    }
                },
            },
            "flow_settings": {
                "input_from_predecessor": ["predictions"],
                "return_value": ["predictions"]
            },
            "job_settings": {
                "asset_name": "advanced_regression",
                "pipeline": "forecast",
                "model_id": "latest",
                "data_id": "latest"
            }
        },
    ],
    "flow_config": {
        "data_settings": {
            # load data
            "flow_return_features": ['df'],
            "flow_return_data": 'predictions',
            "data_index": ['index'],

            # features handling configurations
            "data_handling": {
                'agg_on_columns': ['y_hat'],
                'agg_function_dataframe': 'np.sum'

            },
            "data_output_keys": [
                "missing_values"
            ]
        },
        "model_settings": {
            "variable_to_predict": "target",  # column name of variable to predict
        },
        "job_settings": {
            "asset_name": "flow_regression",
            "pipeline": "forecast_flow"
        }
    }
}
