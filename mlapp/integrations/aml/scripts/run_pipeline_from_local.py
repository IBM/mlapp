from mlapp.integrations.aml.utils.pipeline import run_pipeline_endpoint


def run_script(ws, pipeline_endpoint_name, experiment_name, config_str, pipeline_endpoint_version=None):
    # TODO: perhaps change config str to config path and load it as str
    # publish pipeline endpoint
    run_pipeline_endpoint(ws, pipeline_endpoint_name, experiment_name, config_str,
                          pipeline_version=pipeline_endpoint_version)


# import os
# os.environ['LC_ALL'] = 'en_US.UTF-8'
# os.environ['LANG'] = 'en_US.UTF-8'
# from mlapp.integrations.aml.utils.pipeline import run_pipeline_endpoint
# from mlapp.integrations.aml.utils.workspace import init_workspace
# from config import settings
# import json
#
# pipeline_id = 'pipeline_id'
# experiment_name = 'experiment_name'
# config = {}
# entry_script = 'deployment/aml_target_compute.py'
#
# workspace = init_workspace(
#     settings['aml']['tenant_id'],
#     settings['aml']['subscription_id'],
#     settings['aml']['resource_group'],
#     settings['aml']['workspace_name']
# )
#
# run_pipeline_endpoint(workspace, pipeline_id, experiment_name, json.dumps(config), entry_script)
#
