from azureml.core import Run
from mlapp.main import MLApp
from mlapp.handlers.wrappers.file_storage_wrapper import file_storage_instance
from mlapp.integrations.aml.utils.run_class import load_config_from_string, tag_and_log_run, tag_and_log_outputs
import argparse
from config import settings

from mlapp.managers.flow_manager import FlowManager

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, dest='config', help='configuration')
args = parser.parse_args()
run = Run.get_context()

# pre-processing
config = load_config_from_string(args.config)
tag_and_log_run(config)

# init mlapp
MLApp(settings)

# run config
_, output_ids, output_data = FlowManager(Run.get_context().id, config).run()

# post-processing
tag_and_log_outputs(output_ids)

# post-processing
file_storage_instance.postprocessing()
