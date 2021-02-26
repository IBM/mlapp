import numpy as np

# train configurations
advanced_regression_config = {
    "pipelines_configs": [
        {
            "data_settings": {
                "generate_forecast_data": "index_based",  # normal
                "generate_forecast_data_indices": [35, 42, 57]
            },
            "model_settings": {
                "variable_to_predict": "target"
            },
            "job_settings": {
                "asset_name": "advanced_regression",
                "pipeline": "forecast",
                "model_id": "latest"
            }
        }
    ]
}
