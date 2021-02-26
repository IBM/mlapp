# Configuration Validations

MLApp supports JSON validation using [JSON schema](https://json-schema.org/).  

Validation will be performed automatically by MLApp if a schema file is located in the model's `/config` folder.  

The schema file must be named `<pipeline>_schema` according to the pipeline being run.  
For example: `train_schema.json` or `forecast_schema.json`.

Here's an example of a validation for the `basic_regression` asset `train` pipeline:

```json
{
    "type": "object",
    "properties": {
        "data_settings": {
            "type": "object",
            "properties": {
                "local_file_path": {"type": "string"},
                "variable_to_predict": {"type": "string"},
                "data_handling": {
                    "type": "object",
                    "properties": {
                        "features_for_train": {
                            "type": "array", 
                            "items": {"type": "string"}
                        },
                        "set_features_index": {
                            "type": "array", 
                            "items": {"type": "string"}
                        },
                        "features_to_remove": {
                            "type": "array", 
                            "items": {"type": "string"}
                        },
                        "feature_remove_by_null_percentage": {
                            "type": "number", 
                            "minimum": 0, 
                            "maximum": 1
                        }
                    }
                }
            },
            "required": [
                "local_file_path", 
                "variable_to_predict", 
                "data_handling"
            ]
        },
        "model_settings": {
            "type": "object",
            "properties": {
                "train_percent": {"type": "number", "minimum": 0, "maximum": 1},
                "variable_to_predict": {"type": "string"},
            },
            "required": ["variable_to_predict"]
        },
        "job_settings": {
            "type": "object",
            "properties": {
                "asset_name": {"type": "string"},
                "pipeline": {"type": "string", "pattern": "train"}
            },
            "required": ["asset_name", "pipeline"]
        }
    },
    "required": ["data_settings", "model_settings", "job_settings"]
}
```
