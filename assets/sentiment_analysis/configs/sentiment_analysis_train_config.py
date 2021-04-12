sentiment_analysis_config = {
    "pipelines_configs": [
        {
            # train config
            "data_settings": {
                "sample_data": 5000,
                "local_file_path": "data/yelp_review/train.csv",  # data source
                "variable_to_predict": "target",  # column name of variable to predict
                "text_column": "text",
                "target_column": "target",
                "pre_trained_embeddings": "C:\\Users\\talon\\Downloads\\glove.6B.300d.txt",
                # features handling configurations
                "data_handling": {
                    "features_for_train": [],  # leave empty for selecting all features
                    "set_features_index": [],  # leave empty for indexing by row index
                    # features to remove
                    "features_to_remove": [],
                    "feature_remove_by_null_percentage": 0.3
                }
            },
            # model settings
            "model_settings": {
                "train_percent": 0.8,
                "variable_to_predict": "target", "epochs": 10,
                "batch_size": 5,
                "seed": 1234,
                "to_shuffle": True,
                "embeddings_dim": 300,
                "hidden_layers_size": 100,
                "hidden_dim": 128,
                "num_lstm_layers": 2,
                "dropout": 0.2,
                "bidirectional": True,
                "output_size": 2

            },
            # task settings
            "job_settings": {
                "asset_name": "sentiment_analysis",
                "pipeline": "train"
            }
        }
    ]
}
