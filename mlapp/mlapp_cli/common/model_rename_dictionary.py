rename_dictionary = {
    "base": {
        # key -> file_name: value -> list of dictionaries of words to replace.
        '_data_manager.py': {
            "words": [{"word": 'DataManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_model_manager.py': {
            "words": [{"word": 'ModelManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_train_config.json': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_forecast_config.json': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        }
    },
    "classification": {
        '_data_manager.py': {
            "words": [{"word": 'DataManager', "word_format": 'str_capitalize', 'word_type': 'append-left'},
                      {"word": '_feature_engineering', "word_format": 'str.lower', 'word_type': 'append-left'},
                      {"word": 'from assets.', "word_format": 'str.lower', 'word_type': 'append-right'},
                      {"word": 'FeatureEngineering', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_model_manager.py': {
            "words": [{"word": 'ModelManager', "word_format": 'str_capitalize', 'word_type': 'append-left'},
                      {"word": '_visualizations', "word_format": 'str.lower', 'word_type': 'append-left'},
                      {"word": 'from assets.', "word_format": 'str.lower', 'word_type': 'append-right'}]
        },
        '_train_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_forecast_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*\n*\s*:\s*\n*\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_feature_engineering_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*\n*\s*:\s*\n*\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_reuse_features_and_train_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*\n*\s*:\s*\n*\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_feature_engineering.py': {
            "words": [{"word": 'FeatureEngineering', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_visualizations.py': {
            "words": []
        }
    },
    "crash_course": {
        '_data_manager.py': {
            "words": [{"word": 'DataManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_model_manager.py': {
            "words": [{"word": 'ModelManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_train_config.json': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_forecast_config.json': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        }
    },
    "flow_regression": {
        '_data_manager.py': {
            "words": [{"word": 'DataManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_model_manager.py': {
            "words": [{"word": 'ModelManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_forecast_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        }
    },
    "basic_regression": {
        '_data_manager.py': {
            "words": [{"word": 'DataManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_model_manager.py': {
            "words": [{"word": 'ModelManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_train_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_forecast_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_custom_pipeline_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_feature_engineering_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_train_step_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_reuse_features_and_train_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        }
    },
    "spark_classification": {
        '_data_manager.py': {
            "words": [{"word": 'DataManager', "word_format": 'str_capitalize', 'word_type': 'append-left'},
                      {"word": '_feature_engineering', "word_format": 'str.lower', 'word_type': 'append-left'},
                      {"word": 'from assets.', "word_format": 'str.lower', 'word_type': 'append-right'},
                      {"word": 'FeatureEngineering', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_model_manager.py': {
            "words": [{"word": 'ModelManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_train_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_forecast_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_feature_engineering.py': {
            "words": [{"word": 'FeatureEngineering', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        }
    },
    "spark_regression": {
        '_data_manager.py': {
            "words": [{"word": 'DataManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_model_manager.py': {
            "words": [{"word": 'ModelManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_train_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_forecast_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
    },
    "advanced_regression": {
        '_data_manager.py': {
            "words": [{"word": 'DataManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_model_manager.py': {
            "words": [{"word": 'ModelManager', "word_format": 'str_capitalize', 'word_type': 'append-left'}]
        },
        '_train_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_forecast_config.py': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        },
        '_prediction_accuracy_config.json': {
            'inner_path': 'configs',
            "words": [{"word": '"asset_name": "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*"'},
                      {"word": '"asset_name": \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'"asset_name"\s*:\s*\''},
                      {"word": '\'asset_name\': \'', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*\''},
                      {"word": '\'asset_name\': "', "word_format": 'str.lower', 'word_type': 'append-right',
                       "word_pattern": r'\'asset_name\'\s*:\s*"'},
                      {"word": '_config ', "word_format": 'str.lower', 'word_type': 'append-left'}]
        }
    }
}
