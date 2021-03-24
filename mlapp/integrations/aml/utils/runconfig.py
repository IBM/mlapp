from azureml.core.runconfig import RunConfiguration
from mlapp.integrations.aml.utils.env import create_env_from_requirements


def create_runconfig(aml_compute, env=None):
    # Create a new runconfig object
    aml_run_config = RunConfiguration()

    # Use the aml_compute you created above.
    aml_run_config.target = aml_compute

    if env:
        aml_run_config.environment = env
    else:
        aml_run_config.environment = create_env_from_requirements()

    return aml_run_config
