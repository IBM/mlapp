spark_regression_config = {
    "pipelines_configs": [
        {
            # train config
            "data_settings": {
                "local_file_path": "data/diabetes.csv",
                "variable_to_predict": "target",  # column name of variable to predict
                # features handling configurations
                "data_handling": {
                    "features_for_train": [],  # leave empty for selecting all features
                    "set_features_index": [],  # leave empty for indexing by row index
                    # features to remove
                    "features_to_remove": [
                        "age", "sex", "tc", "ldl",
                        "age_sex", "sex_age",
                        "age_hdl", "hdl_age",
                        "sex_tc", "tc_sex",
                        "sex_ldl", "ldl_sex",
                        "bmi_hdl", "hdl_bmi",
                        "map_hdl", "hdl_map",
                        "tc_hdl", "hdl_tc",
                        "hdl_tch", "tch_hdl",
                        "hdl_ltg", "ltg_hdl",
                        "hdl_glu", "glu_hdl",
                    ],
                    "feature_remove_by_null_percentage": 0.3,
                    # features transformations and missing value strategies
                    "features_handling": {
                        "bmi": {"fillna": "mean", "transformation": ["log1p"]},
                        "hdl": {"fillna": "mean", "transformation": ["log1p"]},
                        "tch": {"fillna": "mean", "transformation": ["log1p"]},
                        "glu": {"fillna": "mean", "transformation": ["log1p"]}
                    },
                    "features_to_bin": [
                        {"name": "tch", "bins": [-0.05, 0.03]},
                        {"name": "hdl", "bins": [0.07]},
                    ],
                    "interactions": True,
                },
                "data_output_keys": {"results": ["missing_values"], "images_files": []}
            },
            # model settings
            "model_settings": {
                "train_percent": 0.8,
                "variable_to_predict": "target",  # column name of variable to predict
                "model_output_keys": {
                    "models_objects": ["model"],
                    "analysis_results": ["scores", "selected_features_names", "model_type_name", "intercept",
                                         "coefficients"]
                }
            },
            # task settings
            "job_settings": {
                "asset_name": "spark_regression",
                "pipeline": "forecast",
                "model_id": "latest",
                "data_id": "latest"
            },
        }
    ]
}
