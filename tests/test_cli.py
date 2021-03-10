import os, time
from mlapp.cli import init
from mlapp.mlapp_cli.mlcp import start as mlcp_start, stop as mlcp_stop, setup as mlcp_setup
from mlapp.mlapp_cli.assets import create as assets_create
from mlapp.mlapp_cli.assets import rename as rename_asset
from mlapp.mlapp_cli.boilerplates import install as boilerplates_install
from mlapp.mlapp_cli.environment import set as envitonment_set, init as envitonment_init
from click.testing import CliRunner
from mlapp.mlapp_cli.common.cli_utilities import get_env, create_directory, create_file, str_to_camelcase
from mlapp.mlapp_cli.common.files import env_file

import unittest


class TestCliMethods(unittest.TestCase):

    def test_init_command(self):
        try:
            runner = CliRunner()
            with runner.isolated_filesystem():
                # directories path
                models_dir_path = 'assets'
                common_dir_path = 'common'
                data_dir_path = 'data'

                # files path
                app_file_path = 'app.py'
                config_file_path = 'config.py'
                run_file_path = 'run.py'
                utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

                result = runner.invoke(init)

                # checks exit code success
                assert result.exit_code == 0

                # checks that directories are created
                assert os.path.exists(models_dir_path)
                assert os.path.exists(common_dir_path)
                assert os.path.exists(data_dir_path)

                # checks that files are created
                assert os.path.exists(app_file_path)
                assert os.path.exists(config_file_path)
                assert os.path.exists(run_file_path)
                assert os.path.exists(utilities_file_path)
        except Exception as e:
            raise e

    def test_init_command_with_all_flags(self):
        try:
            runner = CliRunner()
            with runner.isolated_filesystem():
                # directories path
                models_dir_path = 'assets'
                common_dir_path = 'common'
                data_dir_path = 'data'
                env_dir_path = 'env'
                deployment_dir_path = 'deployment'

                # files path
                app_file_path = 'app.py'
                config_file_path = 'config.py'
                run_file_path = 'run.py'
                gitignore_file_path = '.gitignore'
                dockerignore_file_path = '.dockerignore'
                utilities_file_path = os.path.join(common_dir_path, 'utilities.py')
                env_file_path = os.path.join(env_dir_path, '.env')
                yaml_file_path = os.path.join(deployment_dir_path, 'docker-compose.yaml')

                result = runner.invoke(init, ['-mlcp'])

                # checks exit code success
                assert result.exit_code == 0

                # checks that directories are created
                assert os.path.exists(models_dir_path)
                assert os.path.exists(common_dir_path)
                assert os.path.exists(data_dir_path)
                assert os.path.exists(env_dir_path)
                assert os.path.exists(deployment_dir_path)

                # checks that files are created
                assert os.path.exists(app_file_path)
                assert os.path.exists(config_file_path)
                assert os.path.exists(run_file_path)
                assert os.path.exists(utilities_file_path)
                assert os.path.exists(env_file_path)
                assert os.path.exists(yaml_file_path)
                assert os.path.exists(gitignore_file_path)
                assert os.path.exists(dockerignore_file_path)

                # checks config content env file path
                assert get_env() == env_file_path
        except Exception as e:
            raise e

    def test_mlcp_setup_command(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            env_dir_path = 'env'
            deployment_dir_path = 'deployment'

            # files path
            env_file_path = os.path.join(env_dir_path, '.env')
            yaml_file_path = os.path.join(deployment_dir_path, 'docker-compose.yaml')

            result = runner.invoke(mlcp_setup)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(env_dir_path)
            assert os.path.exists(deployment_dir_path)

            # checks that files are created
            assert os.path.exists(env_file_path)
            assert os.path.exists(yaml_file_path)

            # checks config content env file path
            assert get_env() == env_file_path

    def test_mlcp_start_and_stop_command(self):
        async def start_command(runner):
            runner.invoke(mlcp_start)

        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            env_dir_path = 'env'
            deployment_dir_path = 'deployment'

            # files path
            env_file_path = os.path.join(env_dir_path, '.env')
            yaml_file_path = os.path.join(deployment_dir_path, 'docker-compose.yaml')

            result = runner.invoke(mlcp_setup)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(env_dir_path)
            assert os.path.exists(deployment_dir_path)

            # checks that files are created
            assert os.path.exists(env_file_path)
            assert os.path.exists(yaml_file_path)

            # checks config content env file path
            assert get_env() == env_file_path

            start_command(runner)

            # wait 15 seconds
            time.sleep(15)

            result = runner.invoke(mlcp_stop)

            # checks exit code success
            assert result.exit_code == 0

    def test_environment_init_command(self):

        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            env_dir_path = 'env'

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # creates the env directory if not exists.
            create_directory(directory_name=env_dir_path, include_init=False)

            # checks that directories are created
            assert os.path.exists(env_dir_path)

            test_env_file = 'test_env.env'
            result = runner.invoke(envitonment_init, [test_env_file])

            # checks exit code success
            assert result.exit_code == 0

            # checks config content env file path
            assert get_env() == os.path.join(env_dir_path, test_env_file)

            result = runner.invoke(envitonment_init, [test_env_file])

            # checks exit code success and output already exists
            assert result.exit_code == 0
            assert result.output == 'ERROR: \'' + test_env_file + '\' file already exits.\n'

    def test_environment_set_command(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            env_dir_path = 'env'

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # creates the env directory if not exists.
            create_directory(directory_name=env_dir_path, include_init=False)

            # checks that directories are created
            assert os.path.exists(env_dir_path)

            env_filename = 'test_env.env'
            create_file(file_name=env_filename, path=env_dir_path, content=env_file)

            result = runner.invoke(envitonment_set, [env_filename])

            # checks exit code success
            assert result.exit_code == 0

            # checks config content env file path
            assert get_env() == os.path.join(env_dir_path, env_filename)

    def test_create_asset_command(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            asset_name = 'cli_asset_test'
            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')

            # invoke create model command
            result = runner.invoke(assets_create, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            assert os.path.exists(configs_path)

            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.json'
            forecast_config = asset_name + '_forecast_config.json'
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

    def test_asset_rename_command(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            asset_name = 'cli_asset_test'
            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')

            # invoke create model command
            result = runner.invoke(assets_create, [asset_name])

            # checks exit code success
            assert result.exit_code == 0


            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.json'
            forecast_config = asset_name + '_forecast_config.json'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

            # calling rename command
            asset_name_renamed = 'cli_asset_test_renamed'
            result = runner.invoke(rename_asset, [asset_name, asset_name_renamed])

            # checks exit code success
            assert result.exit_code == 0

            # checking renamed files
            path_model_dir = os.path.join(models_dir_path, asset_name_renamed)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name_renamed + '_data_manager.py'
            model_manager_name = asset_name_renamed + '_model_manager.py'
            train_config = asset_name_renamed + '_train_config.json'
            forecast_config = asset_name_renamed + '_forecast_config.json'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

    def test_asset_rename_command_with_delete_flag(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            asset_name = 'cli_asset_test'
            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')

            # invoke create model command
            result = runner.invoke(assets_create, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.json'
            forecast_config = asset_name + '_forecast_config.json'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

            # calling rename command with delete equals True
            asset_name_renamed = 'cli_asset_test_renamed'
            result = runner.invoke(rename_asset, "" + asset_name + " " + asset_name_renamed + " --delete")

            # checks exit code success
            assert result.exit_code == 0

            # checking renamed files
            new_path_model_dir = os.path.join(models_dir_path, asset_name_renamed)
            configs_path = os.path.join(new_path_model_dir, 'configs')
            data_manager_name = asset_name_renamed + '_data_manager.py'
            model_manager_name = asset_name_renamed + '_model_manager.py'
            train_config = asset_name_renamed + '_train_config.json'
            forecast_config = asset_name_renamed + '_forecast_config.json'
            assert not os.path.exists(path_model_dir)
            assert os.path.exists(new_path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(new_path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(new_path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

    def test_boilerplates_classification(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            asset_name = "classification"
            result = runner.invoke(boilerplates_install, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            visualizations_name = asset_name + '_visualizations.py'
            feature_engineering_name = asset_name + '_feature_engineering.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            feature_engineering_config = asset_name + '_feature_engineering_config.py'
            reuse_features_and_train_config = asset_name + '_reuse_features_and_train_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, feature_engineering_name))
            assert os.path.exists(os.path.join(path_model_dir, visualizations_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))
            assert os.path.exists(os.path.join(configs_path, feature_engineering_config))
            assert os.path.exists(os.path.join(configs_path, reuse_features_and_train_config))

    def test_boilerplates_classification_renamed(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            original_asset_name = "classification"
            asset_name = "classification_renamed"
            result = runner.invoke(boilerplates_install, "" + original_asset_name + " -r " + asset_name)

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            visualizations_name = asset_name + '_visualizations.py'
            feature_engineering_name = asset_name + '_feature_engineering.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            feature_engineering_config = asset_name + '_feature_engineering_config.py'
            reuse_features_and_train_config = asset_name + '_reuse_features_and_train_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, feature_engineering_name))
            assert os.path.exists(os.path.join(path_model_dir, visualizations_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))
            assert os.path.exists(os.path.join(configs_path, feature_engineering_config))
            assert os.path.exists(os.path.join(configs_path, reuse_features_and_train_config))

    def test_boilerplates_advanced_regression(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            asset_name = "advanced_regression"
            result = runner.invoke(boilerplates_install, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            prediction_accuracy_config = asset_name + '_prediction_accuracy_config.json'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))
            assert os.path.exists(os.path.join(configs_path, prediction_accuracy_config))

    def test_boilerplates_advanced_regression_renamed(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            original_asset_name = "advanced_regression"
            asset_name = "advanced_regression_renamed"
            result = runner.invoke(boilerplates_install, "" + original_asset_name + " -r " + asset_name)

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            prediction_accuracy_config = asset_name + '_prediction_accuracy_config.json'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))
            assert os.path.exists(os.path.join(configs_path, prediction_accuracy_config))

    def test_boilerplates_basic_regression(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            asset_name = "basic_regression"
            result = runner.invoke(boilerplates_install, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            custom_pipeline_config = asset_name + '_custom_pipeline_config.py'
            feature_engineering_config = asset_name + '_feature_engineering_config.py'
            reuse_features_and_train_config = asset_name + '_reuse_features_and_train_config.py'
            train_step_config = asset_name + '_train_step_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))
            assert os.path.exists(os.path.join(configs_path, custom_pipeline_config))
            assert os.path.exists(os.path.join(configs_path, feature_engineering_config))
            assert os.path.exists(os.path.join(configs_path, reuse_features_and_train_config))
            assert os.path.exists(os.path.join(configs_path, train_step_config))

    def test_boilerplates_basic_regression_renamed(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            original_asset_name = "basic_regression"
            asset_name = "basic_regression_renamed"
            result = runner.invoke(boilerplates_install, "" + original_asset_name + " -r " + asset_name)

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            custom_pipeline_config = asset_name + '_custom_pipeline_config.py'
            feature_engineering_config = asset_name + '_feature_engineering_config.py'
            reuse_features_and_train_config = asset_name + '_reuse_features_and_train_config.py'
            train_step_config = asset_name + '_train_step_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))
            assert os.path.exists(os.path.join(configs_path, custom_pipeline_config))
            assert os.path.exists(os.path.join(configs_path, feature_engineering_config))
            assert os.path.exists(os.path.join(configs_path, reuse_features_and_train_config))
            assert os.path.exists(os.path.join(configs_path, train_step_config))

    def test_boilerplates_flow_regression(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            asset_name = "flow_regression"
            result = runner.invoke(boilerplates_install, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            flow_config = asset_name + '_forecast_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, flow_config))

    def test_boilerplates_flow_regression_renamed(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            original_asset_name = "flow_regression"
            asset_name = "flow_regression_renamed"
            result = runner.invoke(boilerplates_install, "" + original_asset_name + " -r " + asset_name)

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            flow_config = asset_name + '_forecast_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, flow_config))

    def test_boilerplates_crash_course(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            asset_name = "crash_course"
            result = runner.invoke(boilerplates_install, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.json'
            forecast_config = asset_name + '_forecast_config.json'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

    def test_boilerplates_crash_course_renamed(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            original_asset_name = "crash_course"
            asset_name = "crash_course_renamed"
            result = runner.invoke(boilerplates_install, "" + original_asset_name + " -r " + asset_name)

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.json'
            forecast_config = asset_name + '_forecast_config.json'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

    def test_boilerplates_spark_regression(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            asset_name = "spark_regression"
            result = runner.invoke(boilerplates_install, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

    def test_boilerplates_spark_regression_renamed(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            original_asset_name = "spark_regression"
            asset_name = "spark_regression_renamed"
            result = runner.invoke(boilerplates_install, "" + original_asset_name + " -r " + asset_name)

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

    def test_boilerplates_spark_classification(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            asset_name = "spark_classification"
            result = runner.invoke(boilerplates_install, [asset_name])

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            feature_engineering_name = asset_name + '_feature_engineering.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, feature_engineering_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

    def test_boilerplates_spark_classification_renamed(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # directories path
            models_dir_path = 'assets'
            common_dir_path = 'common'
            data_dir_path = 'data'

            # files path
            app_file_path = 'app.py'
            config_file_path = 'config.py'
            run_file_path = 'run.py'
            utilities_file_path = os.path.join(common_dir_path, 'utilities.py')

            result = runner.invoke(init)

            # checks exit code success
            assert result.exit_code == 0

            # checks that directories are created
            assert os.path.exists(models_dir_path)
            assert os.path.exists(common_dir_path)
            assert os.path.exists(data_dir_path)

            # checks that files are created
            assert os.path.exists(app_file_path)
            assert os.path.exists(config_file_path)
            assert os.path.exists(run_file_path)
            assert os.path.exists(utilities_file_path)

            # invoke create model command
            original_asset_name = "spark_classification"
            asset_name = "spark_classification_renamed"
            result = runner.invoke(boilerplates_install, "" + original_asset_name + " -r " + asset_name)

            # checks exit code success
            assert result.exit_code == 0

            path_model_dir = os.path.join(models_dir_path, asset_name)
            configs_path = os.path.join(path_model_dir, 'configs')
            data_manager_name = asset_name + '_data_manager.py'
            model_manager_name = asset_name + '_model_manager.py'
            feature_engineering_name = asset_name + '_feature_engineering.py'
            train_config = asset_name + '_train_config.py'
            forecast_config = asset_name + '_forecast_config.py'
            assert os.path.exists(path_model_dir)
            assert os.path.exists(configs_path)
            assert os.path.exists(os.path.join(path_model_dir, data_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, model_manager_name))
            assert os.path.exists(os.path.join(path_model_dir, feature_engineering_name))
            assert os.path.exists(os.path.join(configs_path, train_config))
            assert os.path.exists(os.path.join(configs_path, forecast_config))

if __name__ == '__main__':
    unittest.main()
