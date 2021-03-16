import argparse
import os
from mlapp.integrations.aml.utils.constants import MAX_AML_PIPELINE_STEPS, ARG_TYPES, \
    PARSED_ARG_CONFIG, PARSED_ARG_INPUT_DIR, PARSED_ARG_OUTPUT_DIR
from mlapp.utils.general import load_pickle_to_object, save_object_to_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parsed_args = {
        PARSED_ARG_CONFIG: '{}',
        PARSED_ARG_INPUT_DIR: [],
        PARSED_ARG_OUTPUT_DIR: []
    }

    # add arguments for parse
    for i in range(MAX_AML_PIPELINE_STEPS):
        for arg_type in ARG_TYPES:
            parser.add_argument(
                ('--' + arg_type['name'] + str(i)),
                type=str, dest=(arg_type['name'] + str(i)), help=arg_type['help'])

    # parse arguments
    args = parser.parse_args()

    # save parsed arguments
    for i in range(MAX_AML_PIPELINE_STEPS):
        for arg_type in ARG_TYPES:
            if getattr(args, arg_type['name'] + str(i)) is not None:
                if arg_type['type'] == 'str':
                    parsed_args[arg_type['output_key']] = getattr(args, arg_type['name'] + str(i))
                elif arg_type['type'] == 'list':
                    parsed_args[arg_type['output_key']].append(getattr(args, arg_type['name'] + str(i)))

    return parsed_args


def get_job_file_name(directory):
    return os.path.join(directory, 'output.pkl')


def load_job_output(outputs, directory):
    if os.path.exists(os.path.join(os.getcwd(), get_job_file_name(directory))):
        outputs['input_from_predecessor'] += \
            load_pickle_to_object(os.path.join(os.getcwd(), get_job_file_name(directory)))
    else:
        outputs['input_from_predecessor'] += [None]


def is_flow_summary(config):
    return config.get('flow_config', {}) != {}


def flow_setup(current_run_id, config, input_dirs):
    jobs_outputs = {
        'flow_id': current_run_id,
        'input_from_predecessor': [None]
    }
    is_flow_summarizer = is_flow_summary(config)

    if is_flow_summarizer:
        # load all jobs outputs
        for input_dir in input_dirs:
            load_job_output(jobs_outputs, input_dir)
    else:
        # load last job
        if len(input_dirs) > 0:
            jobs_outputs['input_from_predecessor'] = []
            load_job_output(jobs_outputs, input_dirs[-1])

    return jobs_outputs


def flow_postprocess(config, output_data, output_dirs):
    if not is_flow_summary(config):
        for output_dir in output_dirs:
            file_name = get_job_file_name(output_dir)
            file_path = os.path.join(os.getcwd(), file_name)
            os.makedirs(os.path.join(os.getcwd(), output_dir), exist_ok=True)
            save_object_to_pickle(output_data, os.path.join(os.getcwd(), file_path))
