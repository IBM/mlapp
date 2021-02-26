basic_regression_config = {
    "pipelines_configs": [
        {
            # train config
            "data_settings": {
                "local_file_path": "data/diabetes.csv",  # data source
                "variable_to_predict": "target",  # column name of variable to predict
                # features handling configurations
                "data_handling": {
                    "features_for_train": [],  # leave empty for selecting all features
                    "set_features_index": [],  # leave empty for indexing by row index
                    # features to remove
                    "features_to_remove": ["sex"],
                    "feature_remove_by_null_percentage": 0.3
                }
            },
            # model settings
            "model_settings": {
                "train_percent": 0.8,
                "variable_to_predict": "target"  # column name of variable to predict
            },
            "flow_settings": {
                "input_from_predecessor": ["contributer_name"],
                "return_value": ["predictions"]

            },
            # task settings
            "job_settings": {
                "asset_name": "basic_regression",
                "pipeline": "train"
            }
        }
    ]
}
