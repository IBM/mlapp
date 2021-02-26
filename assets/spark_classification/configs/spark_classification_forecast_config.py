spark_classification_config = {
    "pipelines_configs": [{
        "data_settings": {
            "model_name": "breast_cancer",
            "local_data_csvs": [{
                "name": "breast_cancer",
                "path": "data/breast_cancer.csv"
            }],
            "variable_to_predict": "answer",  # column name of variable to predict
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
                "features_to_remove": ["texture error", "area error", "smoothness error",
                                       "compactness error", "concave points error",
                                       "symmetry error", "worst smoothness", "worst compactness",
                                       "worst concavity", "worst concave points", "worst symmetry",
                                       "worst fractal dimension"],
                "feature_remove_by_null_percentage": 0.3,
                "dates_format": "yyyyMMdd",
                "default_missing_value": 0,
                "features_handling": {
                    "mean radius": {"fillna": "mean",
                                    "transformation": ["lambda a: F.pow(a, 2)", "F.sqrt"]},
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
                    {"name": "mean concave points", "bins": [4]},
                    {"name": "mean symmetry", "bins": ['quantile']},
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
                "features_to_convert_to_dummies": ["mean radius", "mean texture", "mean perimeter",
                                                   "mean area",
                                                   "mean smoothness", "mean compactness",
                                                   "mean concavity",
                                                   "mean concave points", "mean symmetry",
                                                   "mean fractal dimension",
                                                   "radius error", "worst area", "worst perimeter",
                                                   "worst texture",
                                                   "worst radius", "fractal dimension error",
                                                   "concavity error",
                                                   "perimeter error"],
                "bin_continuous_features": True,
                "auto_get_dummies": True,
                "max_categories_num": 10,
            },
            "data_output_keys": {"results": ["missing_values", "evaluator_features_mappings"],
                                 "images_files": []},
        },
        "model_settings": {
            "y_label": "POSITIVE",
            "y_name": "answer",
            "custom_tables": {},
            "output_columns_rename": {},
            "model_output_keys": {}
        },
        "job_settings": {
            "asset_name": "spark_classification",
            "model_id": "latest",
            "data_id": "latest",
            "pipeline": "forecast"
        }
    }]
}
