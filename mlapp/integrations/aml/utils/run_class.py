import json
from ast import literal_eval
from azureml.core import Run
from mlapp.integrations.aml.utils.constants import MAX_STRING_LENGTH


def load_config_from_string(config_str):
    try:
        config_str = config_str.decode("utf-8")
    except AttributeError as attrError:
        pass  # message body is string and not bytes

    print("Hello, The following task is consumed:   " + str(config_str))
    try:
        config_str = json.loads(literal_eval(config_str))
    except Exception as first_error:
        try:
            config_str = json.loads(config_str)
        except Exception as second_error:
            print("Error response: " + str('Message not in JSON format'))
            raise second_error

    return config_str


def get_model_register_name(run_id):
    return run_id[:MAX_STRING_LENGTH]


def tag_and_log_run(config):
    run = Run.get_context()
    pipeline_configs = []

    for job in config.get('pipelines_configs', [{}]):
        pipeline_configs.append(job.get('job_settings', {}))

    if 'flow_config' in config:
        pipeline_configs.append(config['flow_config'].get('job_settings', {}))

    for job_config in pipeline_configs:
        for key in ['pipeline', 'asset_name', 'asset_label']:
            value = job_config.get(key)
            if value is not None:
                value_str = value
                if isinstance(value, list):
                    value_str = ','.join(value)
                run.log(key, value=value_str)
                run.tag(key, value=value_str)


def tag_and_log_outputs(run_ids):
    run = Run.get_context()

    for run_id in run_ids:
        run.log("run_id", run_id)
        run.tag("run_id", run_id)

