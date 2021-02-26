import numpy as np

# train configurations
classification_config = {
    "pipelines_configs": [
        {
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
                            },
                        }
                    }
                }
            },
            "job_settings": {
                "asset_name": "classification",
                "pipeline": "reuse_features_and_train",
                "reuse_features_id": "latest",
                "data_id": "latest"
            }
        }
    ]
}
