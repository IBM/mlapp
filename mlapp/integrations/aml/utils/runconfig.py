from azureml.core.runconfig import RunConfiguration


def create_runconfig(aml_compute, env=None):
    # Create a new runconfig object
    aml_run_config = RunConfiguration()

    # Use the aml_compute you created above.
    aml_run_config.target = aml_compute

    if env is not None:
        aml_run_config.environment = env
    else:
        # Enable Docker
        aml_run_config.environment.docker.enabled = True

        # Set Docker base image to the default CPU-based image
        aml_run_config.environment.docker.base_image = "mcr.microsoft.com/azureml/base:0.2.1"

        # Use conda_dependencies.yml to create a conda environment in the Docker image for execution
        aml_run_config.environment.python.user_managed_dependencies = False

    return aml_run_config






