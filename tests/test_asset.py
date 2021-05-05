import os
import traceback
from mlapp.utils.general import get_project_root


from mlapp.main import MLApp

if __name__ == '__main__':
    # -------------------------- Change following config according to your needs ---------------------------------------
    configs = [
        # {
        #     'asset_name': "basic_regression",
        #     'config_path': "assets/basic_regression/configs/basic_regression_train_config.py",
        #     'config_name': "basic_regression_config"
        # },
        # {
        #     'asset_name': "crash_course",
        #     'config_path': "assets/crash_course/configs/crash_course_train_config.json",
        #     'config_name': "crash_course_config"
        # },
        # {
        #     'asset_name': "advanced_regression",
        #     'config_path': "assets/advanced_regression/configs/advanced_regression_train_config.py",
        #     'config_name': "advanced_regression_config"
        # },
        # {
        #     'asset_name': "classification",
        #     'config_path': "assets/classification/configs/classification_train_config.py",
        #     'config_name': "classification_config"
        # },
        # {
        #     'asset_name': "spark_regression",
        #     'config_path': "assets/spark_regression/configs/spark_regression_train_config.py",
        #     'config_name': "spark_regression_config"
        # }
        # {
        #     'asset_name': "spark_classification",
        #     'config_path': "assets/spark_classification/configs/spark_classification_train_config.py",
        #     'config_name': "spark_classification_config"
        # }
        # {
        #     'asset_name': "flow_regression",
        #     'config_path': "assets/flow_regression/configs/flow_regression_forecast_config.py",
        #     'config_name': "flow_config"
        # },
        {
            'asset_name': "sentiment_analysis",
            'config_path': "assets/sentiment_analysis/configs/sentiment_analysis_forecast_config.py",
            'config_name': "sentiment_analysis_config"
        }
    ]
    # ------------------------------------------------------------------------------------------------------------------

    # change directory to project root:
    os.chdir(get_project_root())

    try:
        for config_params in configs:
            kwargs = {'input_from_predecessor': [{'example_input_key': 'example_input_value'}]}

            MLApp({'env_file_path': '.env'}).run_flow(
                config_params['asset_name'], config_params['config_path'], config_params.get('config_name'),
                **kwargs)

        # MLApp({'env_file_path': 'env/.env'}).run_listener()

    except Exception as e:
        print(e)
        traceback.print_exc()
