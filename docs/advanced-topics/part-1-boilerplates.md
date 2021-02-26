# Part 1 - Model Boilerplates

## MLApp's out-of-the-box Classification Boilerplate

MLApp comes with out-of-the-box examples of assets. You can view the available boilerplates with a simple CLI command:

```bash
mlapp boilerplates show
```

Let's install the classification boilerplate giving it a new name: open the terminal in the root of you project and run command:

```bash
mlapp boilerplates install classification --new-name advanced_course
```

!!! note "Installing Model Boilerplates"
    
    You should see a success message that the installation was successful and you should be able to see the asset in your `assets` directory. `--new-name` flag has been used in the command so the name of the asset is going to be **glass_classification**, otherwise it would have been **classification**.

The classification asset example uses the **breast cancer dataset** which should be in your `data` folder after installing the boilerplate.

!!! tip "You can also download the file from the repository"

    1. Click on [data/breast_cancer.csv](https://github.com/ibm/mlapp/blob/master/data/breast_cancer.csv).
    2. Click `Raw` to open the files on github in raw format.
    3. Use your browser to save the current page (Usually by doing `File > Save Page As..`). Make sure to save it as a CSV (extension: .csv)
    4. Save the files as `breast_cancer.csv`.
    5. Move the files to your new project workspace inside the `data` directory.

You can now update your `run.py` file to run the classification.

```json
{
    "asset_name": "advanced_course",
    "config_path": "assets/advanced_course/configs/advanced_course_train_config.py",
    "config_name": "advanced_course_config"
}
```

We recommend exploring the classification boilerplate as it's an example of using the MLApp framework with the best practices.

You can explore the config file `assets > advanced_course > configs > advanced_course_train_config.py`. It's an example of a config for the classification asset boilerplate with useful comments.

## Run configuration for the glass dataset

To run the created Advanced Course asset for the glass dataset (which was used in the [Crash Course](/crash_course/introduction)), we will use a new configuration that is adjusted to the glass dataset and it's features without applying any changes the code.

We have prepared this configuration for the glass dataset.

Create a new config file at `assets > advanced_course > configs > advanced_course_glass_train_config.json`

Insert into the file the following content:

```json
{
  "pipelines_configs": [
    {
      "data_settings": {
        "local_data_csvs": [{ "name": "glass", "path": "data/glass.csv"}],
        "variable_to_predict": "Type",
        "data_handling": {
          "y_variable": {
            "type": "multi",
            "label_to_predict": ["1", "2", "3", "4", "5", "6", "7"]
          },
          "feature_remove_by_null_percentage": 0.3,
          "features_handling": {},
          "features_to_bin": [
            {"name": "RI", "bins": [1.516523, 1.519157]},
            {"name": "Na", "bins": [12.9075, 13.825]},
            {"name": "Mg", "bins": [2.115000, 3.600000]},
            {"name": "Al", "bins": [1.19, 1.63]},
            {"name": "Si", "bins": [72.28, 73.0875]},
            {"name": "K", "bins": [0.1225, 0.61]},
            {"name": "Ca", "bins": [8.24, 9.1725]},
            {"name": "Ba", "bins": [0.025]},
            {"name": "Fe", "bins": [0, 0.1]}
          ],
          "use_evaluator": false
        }
      },
      "model_settings": {
        "variable_to_predict": "Type",
        "down_sample_method": {"flag": false, "n_samples": 100, "seed": 1500},
        "train_percent": 0.8,
        "predict_proba_threshold": [0.05, 0.5, 0.95],
        "auto_ml": {
          "feature_selection": [
              {
                  "method": "SelectKBest",
                  "params": {
                      "k": 7,
                      "score_func": "chi2"
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
          "estimator": "multi_class",
          "multi_class": {
              "models": ["MultiLogistic", "ExtraTreeClassifier", "KnearestNeighbors", "RandomForest"]
          }
      }
      },
      "job_settings": {
        "asset_name": "advanced_course",
        "pipeline": "train"
      }
    }
  ]
}
```

You can now run the advanced course asset via `run.py` with the new glass dataset configuration:

```json
{
    "asset_name": "advanced_course",
    "config_path": "assets/advanced_course/configs/advanced_course_glass_train_config.json",
    "config_name": "advanced_course_config"
}
```

That's it, now you have run one of our Boilerplates. Next steps you can explore the other Boilerplates available.

When starting a development of new asset it is recommended to start with one of our Boilerplates!