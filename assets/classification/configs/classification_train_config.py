# train configurations
classification_config = {
    "pipelines_configs": [
        {
            "data_settings": {
                "model_name": "breast_cancer",
                "local_data_csvs": [{
                    "name": "breast_cancer",
                    "path": "data/breast_cancer.csv"
                }],
                # features handling configurations
                "data_handling": {
                    # y variable feature engineering
                    "y_variable": {
                        "type": "binary",  # binary/multi/continuous
                        "categories_labels": ["NEGATIVE", "POSITIVE"],  # category labels
                        "continuous_to_category_bins": [-1, 1, 2],
                        # bins values in case there is a need to cut the y to categories
                        "label_to_predict": ["POSITIVE"]  # target label to predict
                    },
                    "features_for_filter": {},  # leave empty for not filter
                    "features_for_train": None,  # leave empty for selecting all features
                    "set_features_index": None,  # leave empty for indexing by row index
                    "features_to_remove": ["texture error", "area error", "smoothness error", "compactness error",
                                           "concave points error", "symmetry error", "worst smoothness",
                                           "worst compactness", "worst concavity", "worst concave points",
                                           "worst symmetry", "worst fractal dimension"],
                    "feature_remove_by_null_percentage": 0.3,
                    "dates_format": ["%d/%m/%Y", "%Y-%m-%d"],
                    "default_missing_value": 0,
                    "features_handling": {
                        "mean radius": {"fillna": "np.mean", "transformation": ["np.square", "np.sqrt"]},
                        "radius error": {"fillna": 0, "transformation": []}
                    },
                    "features_interactions": [],
                    "dates_transformation": {"extraction_date": "20180430", "columns": []},
                    "features_to_bin": [
                        {"name": "mean radius", "bins": [12.3, 15.3]},
                        {"name": "mean texture", "bins": [15, 23]},
                        {"name": "mean perimeter", "bins": [72, 109]},
                        {"name": "mean area", "bins": [361, 886]},
                        {"name": "mean smoothness", "bins": [0.074, 0.11]},
                        {"name": "mean compactness", "bins": [0.047, 0.137, 0.228]},
                        {"name": "mean concavity", "bins": [0.023, 0.12]},
                        {"name": "mean concave points", "bins": [0.025]},
                        {"name": "mean symmetry", "bins": [0.142, 0.2, 0.26]},
                        {"name": "mean fractal dimension", "bins": [0.0518, 0.0541, 0.0742, 0.0827]},
                        {"name": "radius error", "bins": [0.19, 0.56, 0.83]},
                        {"name": "worst area", "bins": [500, 1050]},
                        {"name": "worst perimeter", "bins": [85, 120]},
                        {"name": "worst texture", "bins": [16.6, 42]},
                        {"name": "worst radius", "bins": [12.5, 18]},
                        {"name": "fractal dimension error", "bins": [0.006, 0.0135]},
                        {"name": "concavity error", "bins": [0.011, 0.082, 0.15]},
                        {"name": "perimeter error", "bins": [1.3, 5]}
                    ],
                    "action_for_continuous_features": "auto_bin",  # auto_bin/keep_as_is/remove
                    # evaluator properties
                    "evaluator_settings": {
                        "filter_evaluator_threshold": 0.05,
                        "store_evaluator_features": True
                    }
                }
            },
            "model_settings": {
                "variable_to_predict": "answer",
                "down_sample_method": {
                    "flag": False,
                    "n_samples": 100,
                    "seed": 1500
                },
                "train_percent": 0.8,
                "predict_proba_threshold": [0.05, 0.5, 0.95],
                "auto_ml": {
                    "feature_selection": [
                        {
                            "method": "SelectKBest",
                            "params": {
                                'k': 7,
                                'score_func': 'chi2'
                            }
                        }
                    ],
                    "estimator": "binary",
                    "binary": {
                        "models": ["XGBoostClassifier", "Logistic", "ExtraTreeClassifier"],
                        "fixed_params": {
                            'Logistic': {
                                'solver': 'liblinear',
                                'max_iter': 500
                            },
                            'ExtraTreeClassifier': {
                                'min_samples_leaf': 4,
                                'max_depth': 10,
                                'class_weight': 'balanced'
                            }
                        },
                        "hyper_params": {
                            'Logistic': {
                                'C': 'list(np.linspace(0.01, 4, 15))',
                                'penalty': ['l1', 'l2'],
                                'class_weight': ['balanced', None],
                                'fit_intercept': [True],
                            },
                            'XGBoostClassifier': {
                                'max_depth': [3, 7, 10],
                                'learning_rate': 'list(np.linspace(0.001, 0.02, 3))',
                                'n_estimators': [300, 500, 700],
                                'min_child_weight': [3, 10],
                            },
                            'ExtraTreeClassifier': {
                                'n_estimators': [10, 50, 200],
                                'min_samples_split': [2, 10],
                            }
                        }
                    }
                }
            },
            "job_settings": {
                "asset_name": "classification",
                "pipeline": "train",
            }
        }
    ]
}
