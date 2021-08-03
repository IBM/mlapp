from azureml.pipeline.core import PublishedPipeline
from mlapp.integrations.aml.utils.deploy import get_best_model_in_experiment
from mlapp.integrations.aml.utils.pipeline import run_pipeline_endpoint
import json


def run_script(ws, pipeline_endpoint_name, experiment_name, config_str, pipeline_endpoint_version=None):
    # TODO: perhaps change config str to config path and load it as str
    # publish pipeline endpoint
    run_pipeline_endpoint(ws, pipeline_endpoint_name, experiment_name, config_str,
                          pipeline_version=pipeline_endpoint_version)


def run_forecast(ws, asset_name, train_experiment_name, score_metric, greater_is_better, config_str, experiment_name):
    run_id, pipeline_id = get_best_model_in_experiment(
        ws, train_experiment_name, asset_name, None, score_metric, greater_is_better, return_pipeline_id=True)

    config_dict = json.loads(config_str)
    config_dict['pipelines_configs'][0]['job_settings']['model_id'] = run_id

    run_pipeline_endpoint(ws, pipeline_id, experiment_name, json.dumps(config_dict))


def run_model_drift(ws, asset_name, train_experiment_name, score_metric, greater_is_better, config_str, experiment_name):
    run_id, pipeline_id = get_best_model_in_experiment(
        ws, train_experiment_name, asset_name, None, score_metric, greater_is_better, return_pipeline_id=True)

    config_dict = json.loads(config_str)
    config_dict['pipelines_configs'][0]['job_settings']['model_id'] = run_id

    run_pipeline_endpoint(ws, pipeline_id, experiment_name, config_str)


def run_train(ws, experiment_name, config_str):
    latest_version = 1
    pipeline_id = None
    for pipeline in PublishedPipeline.get_all(ws, True):
        version = int(pipeline.name.split('_')[-2][1:] + pipeline.name.split('_')[-1])  # format: vYYYYmmDD_n
        if version >= latest_version:
            pipeline_id = pipeline.id

    run_pipeline_endpoint(ws, pipeline_id, experiment_name, config_str)


# import os
# os.environ['LC_ALL'] = 'en_US.UTF-8'
# os.environ['LANG'] = 'en_US.UTF-8'
# from mlapp.integrations.aml.utils.pipeline import run_pipeline_endpoint
# from mlapp.integrations.aml.utils.workspace import init_workspace
# from config import settings
# import json
#
# experiment_name = 'experiment-name'
# asset_name = 'asset_name'
# score_metric = 'score_metric'
# greater_is_better = True
# config_str = '{ "pipelines_configs": [ { "data_settings": {}, "model_settings": {}, "job_settings": {} } ] }'
#
# workspace = init_workspace(
#     settings['aml']['tenant_id'],
#     settings['aml']['subscription_id'],
#     settings['aml']['resource_group'],
#     settings['aml']['workspace_name']
# )
#
#
#
# run_id, pipeline_id = get_best_model_in_experiment(
#     workspace, experiment_name, asset_name, None, score_metric, greater_is_better, return_pipeline_id=True)
#
# config_dict = json.loads(config_str)
# config_dict['pipelines_configs'][0]['job_settings']['model_id'] = run_id
# run_pipeline_endpoint(workspace, pipeline_id, experiment_name, json.dumps(config_dict))
