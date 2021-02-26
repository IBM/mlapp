basic_regression_config = {
    "pipelines_configs": [
        # train config
        {
            "data_settings": {
                "local_file_path": "data/diabetes.csv",  # data source
                "variable_to_predict": "target",  # column name of variable to predict
            },
            # forecast config
            "model_settings": {
                "local_file_path": "data/diabetes.csv",   # data source
                "variable_to_predict": "target"
            },
            # job settings
            "job_settings": {
                "asset_name": "basic_regression",
                "model_id": "latest",
                "pipeline": "forecast"
            }
        }
    ]
}