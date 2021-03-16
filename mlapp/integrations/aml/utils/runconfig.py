from azureml.core.runconfig import RunConfiguration
from azureml.core.environment import DEFAULT_CPU_IMAGE
from azureml.core import Environment


def create_runconfig(aml_compute, env=None):
    # Create a new runconfig object
    aml_run_config = RunConfiguration()

    # Use the aml_compute you created above.
    aml_run_config.target = aml_compute

    if env is not None:
        aml_run_config.environment = env
    else:
        aml_run_config.environment = Environment.from_pip_requirements(name='mlapp', file_path='requirements.txt')

        # Enable Docker
        aml_run_config.environment.docker.enabled = True

        # Set Docker base image to the default CPU-based image
        aml_run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE

        # Use conda_dependencies.yml to create a conda environment in the Docker image for execution
        aml_run_config.environment.python.user_managed_dependencies = False

    return aml_run_config
