import os
import click
import requests

from mlapp.mlapp_cli.common.cli_utilities import create_directory, create_file
from mlapp.mlapp_cli.common.files import app_file, utilities_file, empty_config_file, default_config_file, \
    env_file, docker_compose_file, run_file, gitignore_file, dockerignore_file
try:
    from mlapp.integrations.aml.cli import _setup_aml as setup_aml_env
except:
    setup_aml_env = None

init_files_directories = ['assets', 'common', 'data', 'deployment', 'env']


def init_command(ml_control_panel, azure_machine_learning, is_gitignore, is_dockerignore, is_force_init):
    if not is_force_init:
        is_initiated = False
        exsiting_files = os.listdir(os.getcwd())
        for f in exsiting_files:
            full_path = os.path.join(os.getcwd(), f)
            if (os.path.isdir(full_path) and f in init_files_directories) or '.py' in f:
                is_initiated = True
                break

        if is_initiated:
            click.secho(
                "ERROR: your project is not empty.\nHint: you can use 'mlapp init --force' option to force init (caution: force may override exsiting files).", fg='red')
            return

    # creates the assets directory if not exists.
    create_directory(directory_name='assets')

    # creates the common directory if not exists.
    create_directory(directory_name='common')

    # creates the data directory if not exists.
    create_directory(directory_name='data', include_init=False)

    # generates app file template
    app_file_content = app_file

    # generates run file template
    run_file_content = run_file

    # generates utilities file template
    utilities_file_content = utilities_file

    config_file_content = empty_config_file
    # create all files from templates
    create_file(file_name='app.py', content=app_file_content)
    create_file(file_name='run.py', content=run_file_content)
    create_file(file_name='utilities.py', path='common', content=utilities_file_content)
    create_file(file_name='config.py', content=config_file_content)
    create_file(file_name='requirements.txt')

    if ml_control_panel:
        # creates the env directory if not exists.
        create_directory(directory_name='env', include_init=False)

        # creates the deployment directory if not exists.
        create_directory(directory_name='deployment', include_init=False)

        docker_compose_file_content = docker_compose_file

        env_file_content = env_file
        # creates files
        create_file(file_name='docker-compose.yaml', path='deployment', content=docker_compose_file_content)
        create_file(file_name='.env', path='env', content=env_file_content)

        # edit content of config.py file, set env new file
        default_env_filename = ""
        config_file_content = default_config_file.replace("<FILENAME>", default_env_filename)
        create_file(file_name='config.py', content=config_file_content)

    if azure_machine_learning:
        if setup_aml_env is not None:
            setup_aml_env()
        else:
            click.secho("Warning: 'azureml sdk is not installed in your environment. please install it and run 'mlapp aml setup' to complete the init operation.", fg='red')

    if not is_gitignore:
        if not os.path.exists(os.path.join(os.getcwd(), '.gitignore')) or is_force_init:
            try:
                # attempt to get gitignore from github
                response = requests.get('https://raw.githubusercontent.com/github/gitignore/master/Python.gitignore')
                if response.status_code!=200:
                    raise RuntimeError('attempted and failed to fetch .gitignore from github.'
                                       ' Will use default')
                # append ml app gitignore
                github_gitignore_file = response.content.decode()
                github_gitignore_file += f'\n{gitignore_file}'

                # dump
                create_file('.gitignore', content=github_gitignore_file)
            except:
                create_file('.gitignore', content=gitignore_file)
        else:
            click.secho("Error: '.gitignore' already exists.", fg='red')

    if not is_dockerignore:
        if not os.path.exists(os.path.join(os.getcwd(), '.dockerignore')) or is_force_init:
            create_file('.dockerignore', content=dockerignore_file)
        else:
            click.secho("Error: '.dockerignore' already exists.", fg='red')
