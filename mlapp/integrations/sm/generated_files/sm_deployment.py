from mlapp import MLApp
from mlapp.managers.flow_manager import FlowManager
from config import settings
from sagemaker_training import environment


def inject_hyperparam_ranges(config, hyperparam_ranges):
    for i in range(len(config)):
        if 'model_settings' not in config[i]:
            config[i]['model_settings'] = {}
        config[i]['model_settings']['hyperparameter_ranges'] = hyperparam_ranges


if __name__ == '__main__':
    # env handling
    env = environment.Environment()
    settings['local_storage_path'] = env.model_dir

    # config handling
    default_config = '{"pipelines_configs": [{"data_settings": {}, "model_settings": {}, "job_settings": {}}]}'
    config = env.hyperparameters.get('config', default_config)

    inject_hyperparam_ranges(config, {k: v for k, v in env.hyperparameters.items() if k.startswith('MLAPP_')})

    print(f"[ MLAPP ] Received config: {config}")

    # run job
    mlapp = MLApp(settings)
    _, output_ids, output_data = FlowManager(env.job_name, config).run()


