basic_regression_config = {
    "pipelines_configs": [
        {
            # train config
            "data_settings": {},
            # model settings
            "model_settings": {
                "train_percent": 0.8,
                "variable_to_predict": "target"  # column name of variable to predict
            },
            # task settings
            "job_settings": {
                "asset_name": "basic_regression",
                "reuse_features_id": "latest",
                "pipeline": "reuse_features_and_train"
            }
        }
    ]
}
