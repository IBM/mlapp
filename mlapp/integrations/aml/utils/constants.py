MAX_STRING_LENGTH = 31
OUTPUTS_FOLDER = "outputs"
AML_MLAPP_FOLDER = "mlapp"
MAX_AML_PIPELINE_STEPS = 10
CONFIG_ARGUMENT = 'config'
INPUT_DATA_ARGUMENT = 'input'
OUTPUT_DATA_ARGUMENT = 'output'
ARG_TYPES = [
    {'name': CONFIG_ARGUMENT, 'output_key': 'config_str', 'help': 'configuration', 'type': 'str'},
    {'name': INPUT_DATA_ARGUMENT, 'output_key': 'input_dirs', 'help': 'input directory', 'type': 'list'},
    {'name': OUTPUT_DATA_ARGUMENT, 'output_key': 'output_dirs', 'help': 'output directory', 'type': 'list'}
]
OUTPUT_PATH_ON_COMPUTE = 'results'
DATA_REFERENCE_NAME = 'results_dir'
PARSED_ARG_CONFIG = 'config_str'
PARSED_ARG_INPUT_DIR = 'input_dirs'
PARSED_ARG_OUTPUT_DIR = 'output_dirs'
