import numpy as np

# train configurations
advanced_regression_config = {
    "pipelines_configs": [
        {
            "data_settings": {
                # load data
                "data_sources": {
                    # local file
                    "local": {
                        "file_paths": ["data/diabetes.csv"]  # local files
                    },
                    # database
                    # "db": {
                    #     "query": "SELECT * FROM diabetes",  # query to fetch data-m ,
                    #     "args": []  # arguments for query
                    # },
                    # s3 file store
                    # "s3": {
                    #     "buckets": {
                    #         "input": {  # bucket name
                    #             "file_keys": []  # file keys in bucket
                    #         }
                    #     }
                    # }
                },
                # features handling configurations
                "data_handling": {
                    "features_for_train": [],  # leave empty for selecting all features
                    "set_features_index": [],  # leave empty for indexing by row index
                    # features to remove
                    "features_to_remove": [
                        "age", "sex", "tc", "ldl", "age_sex", "sex_age", "age_hdl", "hdl_age", "sex_tc", "tc_sex",
                        "sex_ldl", "ldl_sex", "bmi_hdl", "hdl_bmi", "map_hdl", "hdl_map", "tc_hdl", "hdl_tc",
                        "hdl_tch", "tch_hdl", "hdl_ltg", "ltg_hdl", "hdl_glu", "glu_hdl",

                    ],
                    "feature_remove_by_null_percentage": 0.3,
                    # features transformations and missing value strategies
                    "features_handling": {
                        "bmi": {"fillna": "np.mean", "transformation": ["np.square"]},
                        "hdl": {"fillna": "np.mean", "transformation": ["np.square"]},
                        "tch": {"fillna": "np.mean", "transformation": ["np.square"]},
                        "glu": {"fillna": "np.mean", "transformation": ["np.square"]}
                    },
                    "features_to_bin": [
                        {"name": "tch", "bins": [-0.05, 0.03]},
                        {"name": "hdl", "bins": [0.07]},
                    ],
                    "interactions": True,
                },
            },
            "model_settings": {
                "train_percent": 0.8,
                "variable_to_predict": "target",
                "auto_ml": {
                    "feature_selection": [
                        {
                            "method": "SelectKBest",
                            "params": {
                                'k': 7,
                                'score_func': 'f_regression'
                            }
                        },
                        {
                            "method": "RFE",
                            "params": {
                                "estimator": "SVR(kernel='linear')",
                                "n_features_to_select": 5,
                                "step": 1
                            }
                        },
                        {
                            "method": "AllFeatures"
                        }
                    ],
                    "estimators": {
                        "linear": {
                            "models": ["Lasso", "Ridge"]
                        },
                        "non_linear": {
                            "models": ["RandomForest",  "ExtraTree"],
                            "hyper_params": {
                                "RandomForest": {
                                    "max_depth": [1, 3],
                                    "n_estimators": [10, 50]
                                }
                            }
                        }
                    }
                }
            },
            "job_settings": {
                "asset_name": "advanced_regression",
                "pipeline": "train",
            }
        }
    ]
}