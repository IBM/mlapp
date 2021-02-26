from azureml.core import Environment


def get_mlapp_environment(workspace, env_name, version=None):
    return Environment.get(workspace=workspace, name=env_name, version=version)


def create_or_update_mlapp_env(workspace, requirements_path, wheel_path, env_name):
    """
    Usage:
    ws = init_workspace()
    create_mlapp_environment(
            workspace=ws,
            requirements_path='../../../requirements.txt',
            wheel_path='./../../dist/mlapp-2.0.0-py3-none-any.whl',
            env_name='mlapp')
    """

    # get or create environment and add requirements.txt file
    try:
        restored_env = Environment.get(workspace=workspace, name=env_name)
        new_env = restored_env.from_pip_requirements(name=env_name, file_path=requirements_path)
    except Exception as e:
        new_env = Environment.from_pip_requirements(name=env_name, file_path=requirements_path)

    # settings for environment
    new_env.docker.enabled = True
    new_env.python.user_managed_dependencies = False

    # add private package
    whl_url = Environment.add_private_pip_wheel(workspace, wheel_path, exist_ok=False)
    new_env.python.conda_dependencies.add_pip_package(whl_url)

    # build and register environment
    new_env = new_env.register(workspace)
    build_env_run = new_env.build(workspace)
    build_env_run.wait_for_completion(show_output=False)
    print(build_env_run.log_url)
    print(build_env_run.status)


def display_mlapp_environments(workspace, name='mlapp'):
    envs = Environment.list(workspace=workspace)

    for env in envs:
        if env.startswith(name):
            print("Name", env)
            print("packages", envs[env].python.conda_dependencies.serialize_to_string())