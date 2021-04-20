import shutil
import os
from mlapp.utils.general import get_project_root
import unittest
import warnings
from mlapp.config import settings
from mlapp.main import MLApp
os.chdir(get_project_root())


class TestAssets(unittest.TestCase):
    # --------------------------------------------- Test options -------------------------------------------------------
    output_folder = 'test_output/'
    delete_output_at_finish = False
    env_path = '.env'
    # ------------------------------------------------------------------------------------------------------------------

    def setUp(self):
        warnings.filterwarnings("ignore", category=ResourceWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def tearDown(self) -> None:
        # deleting all files
        if TestAssets.delete_output_at_finish:
            for filename in os.listdir(TestAssets.output_folder):
                # full path file
                full_path_file_name = os.path.join(TestAssets.output_folder, filename)

                if os.path.isdir(full_path_file_name):
                    shutil.rmtree(full_path_file_name)
                else:
                    os.remove(full_path_file_name)
            os.rmdir(TestAssets.output_folder)

    models_available = {
        'basic_regression': {
            'train': {
                'model_name': "basic_regression",
                'config_path': "assets/basic_regression/configs/basic_regression_train_config.yaml",
                'config_name': "basic_regression_config"
            },
            'forecast': {
                'model_name': "basic_regression",
                'config_path': "assets/basic_regression/configs/basic_regression_forecast_config.yaml",
                'config_name': "basic_regression_config"
            },
            'custom': {
                'model_name': "basic_regression",
                'config_path': "assets/basic_regression/configs/basic_regression_custom_pipeline_config.yaml",
                'config_name': "basic_regression_config"
            }

        },
        'advanced_regression': {
            'train': {
                'model_name': "advanced_regression",
                'config_path': "assets/advanced_regression/configs/advanced_regression_train_config.yaml",
                'config_name': "advanced_regression_config"
            },
            'forecast': {
                'model_name': "advanced_regression",
                'config_path': "assets/advanced_regression/configs/advanced_regression_forecast_config.yaml",
                'config_name': "advanced_regression_config"
            }
        },
        'classification': {
            'train': {
                'model_name': "classification",
                'config_path': "assets/classification/configs/classification_train_config.yaml",
                'config_name': "classification_config"
            },
            'forecast': {
                'model_name': "classification",
                'config_path': "assets/classification/configs/classification_forecast_config.yaml",
                'config_name': "classification_config"
            },
            'feature_engineering': {
                'model_name': "classification",
                'config_path': "assets/classification/configs/classification_feature_engineering_config.yaml",
                'config_name': "classification_config"
            },
            'reuse_features_and_train': {
                'model_name': "classification",
                'config_path': "assets/classification/configs/classification_reuse_features_and_train_config.yaml",
                'config_name': "classification_config"
            }
        },
        'crash_course': {
            'train': {
                'model_name': "crash_course",
                'config_path': "assets/crash_course/configs/crash_course_train_config.yaml",
            },
            'forecast': {
                'model_name': "crash_course_basic",
                'config_path': "assets/crash_course/configs/crash_course_forecast_config.yaml",
            }
        },
        'spark_regression': {
            'train': {
                'model_name': "spark_regression",
                'config_path': "assets/spark_regression/configs/spark_regression_train_config.yaml",
                'config_name': "spark_regression_config"
            },
            'forecast': {
                'model_name': "spark_regression",
                'config_path': "assets/spark_regression/configs/spark_regression_forecast_config.yaml",
                'config_name': "spark_regression_config"
            }
        },
        'spark_classification': {
            'train': {
                'model_name': "spark_classification",
                'config_path': "assets/spark_classification/configs/spark_classification_train_config.yaml",
                'config_name': "spark_classification_config"
            },
            'forecast': {
                'model_name': "spark_classification",
                'config_path': "assets/spark_classification/configs/spark_classification_forecast_config.yaml",
                'config_name': "spark_classification_config"
            }
        },
        'flow_regression': {
            'forecast_flow': {
                'model_name': "flow_regression",
                'config_path': "assets/flow_regression/configs/flow_regression_forecast_config.yaml",
                'config_name': "flow_regression_forecast_config"
            }
        }
    }

    def test_basic_regression(self):
        self._inner_func(['basic_regression'])

    def test_advanced_regression(self):
        self._inner_func(['advanced_regression'])

    def test_classification(self):
        self._inner_func(['classification'])

    def test_crash_course(self):
        self._inner_func(['crash_course'])

    def test_flow_regression(self):
        self._inner_func(['flow_regression'])

    def test_spark_regression(self):
        try:
            import pyspark
            self._inner_func(['spark_regression'])
        except ImportError:
            warnings.warn("Missing pyspark library installation for testing `spark_regression`")

    def test_spark_classification(self):
        try:
            import pyspark
            self._inner_func(['spark_classification'])
        except ImportError:
            warnings.warn("Missing pyspark library installation for testing `spark_classification`")

    @staticmethod
    def _inner_func(models_to_test):
        try:
            try:
                import pyspark
                os.environ['LOCAL-SPARK_MLAPP_SERVICE_TYPE'] = 'spark'
                os.environ['LOCAL-SPARK_MAIN_SPARK'] = 'true'
            except ImportError:
                pass

            settings['local_storage_path'] = TestAssets.output_folder
            mlapp = MLApp({'env_file_path': TestAssets.env_path})

            if not os.path.exists(TestAssets.output_folder):
                os.makedirs(TestAssets.output_folder)

            for model_key in models_to_test:

                print('###############################################################################################')
                print('                              ' + model_key)
                print('###############################################################################################')

                model = TestAssets.models_available[model_key]

                for key in ['custom', 'data_explore', 'train', 'forecast',
                            'forecast_flow', 'feature_engineering', 'reuse_features_and_train']:
                    if key in model:
                        mlapp.run_flow(
                            model[key]['model_name'], model[key]['config_path'], model[key].get('config_name'))

            print(f"'{models_to_test}' test is done.")
        except Exception as e:
            raise e


if __name__ == '__main__':
    unittest.main()
