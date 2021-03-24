import os
from azureml.core import Environment
from azureml.core.environment import DEFAULT_CPU_IMAGE


def get_mlapp_environment(workspace, env_name, version=None):
    return Environment.get(workspace=workspace, name=env_name, version=version)


def create_env_from_requirements(file_path='requirements.txt', name='mlapp', endpoint=False):
    env = Environment.from_pip_requirements(name=name, file_path=file_path)

    # Enable Docker
    env.docker.enabled = True

    # Set Docker base image to the default CPU-based image
    path_to_dockerfile = os.path.join(os.getcwd(), 'deployment', 'Dockerfile')
    if os.path.exists(path_to_dockerfile):
        # Set custom Docker image
        env.docker.base_image = None
        env.docker.base_dockerfile = path_to_dockerfile
    else:
        # Set Docker base image to the default CPU-based image
        env.docker.base_image = DEFAULT_CPU_IMAGE

    # Use conda_dependencies.yml to create a conda environment in the Docker image for execution
    env.python.user_managed_dependencies = False

    if endpoint:
        env.inferencing_stack_version = 'latest'

    return env


def display_mlapp_environments(workspace, name='mlapp'):
    envs = Environment.list(workspace=workspace)

    for env in envs:
        if env.startswith(name):
            print("Name", env)
            print("packages", envs[env].python.conda_dependencies.serialize_to_string())