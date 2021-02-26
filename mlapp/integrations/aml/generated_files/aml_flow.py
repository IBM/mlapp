from azureml.core import Run
from mlapp.main import MLApp
from mlapp.handlers.wrappers.file_storage_wrapper import file_storage_instance
from mlapp.integrations.aml.utils.flow import parse_args, flow_setup, flow_postprocess
from mlapp.integrations.aml.utils.constants import PARSED_ARG_CONFIG, PARSED_ARG_INPUT_DIR, PARSED_ARG_OUTPUT_DIR
from mlapp.managers.flow_manager import FlowManager
from mlapp.integrations.aml.utils.run_class import load_config_from_string, tag_and_log_run, tag_and_log_outputs
from config import settings

# parsing arguments
parsed_args = parse_args()

# pre-processing
config = load_config_from_string(parsed_args[PARSED_ARG_CONFIG])
tag_and_log_run(config)

# current run identification
current_run = Run.get_context()

# init mlapp
MLApp(settings)

# flow setup
jobs_outputs = flow_setup(current_run.id, config, parsed_args[PARSED_ARG_INPUT_DIR])

# run config
_, output_ids, output_data = FlowManager(current_run.id, config, **jobs_outputs).run()

# post-processing
tag_and_log_outputs(output_ids)
file_storage_instance.postprocessing()
flow_postprocess(config, output_data, parsed_args[PARSED_ARG_OUTPUT_DIR])
