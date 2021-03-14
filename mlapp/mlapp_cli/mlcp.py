import os, click
from mlapp.mlapp_cli.common.cli_utilities import create_file, create_directory, set_env, get_available_port
from mlapp.mlapp_cli.common.files import docker_compose_file, env_file, default_config_file, vue_env_config_file, init_sql_file
from mlapp.mlapp_cli.cli_help import cli_mlcp_help


@click.group("mlcp")
def commands():
    """
    ML App MLCP Command

    Use it to setup and run ML App MLCP locally on your machine.

    """
    pass


@commands.command("setup", help=cli_mlcp_help.get('setup', 'setup mlcp on your machine'))
def setup():
    # creates the env directory if not exists.
    create_directory(directory_name='env', include_init=False)

    # creates the deployment directory if not exists.
    create_directory(directory_name='deployment', include_init=False)

    docker_compose_file_content = docker_compose_file
    init_sql_file_content = init_sql_file
    env_file_content = env_file

    # creates files
    create_file(file_name='docker-compose.yaml', path='deployment', content=docker_compose_file_content)
    create_file(file_name='init.sql', path='deployment', content=init_sql_file_content)
    #create_file(file_name='env-config.js', path='deployment', content=vue_env_config_file)

    create_file(file_name='.env', path='env', content=env_file_content)

    # edit content of config.py file, set env new file
    default_env_filename = ''
    config_file_content = default_config_file.replace("<FILENAME>", default_env_filename)
    create_file(file_name='config.py', content=config_file_content)


@commands.command("start", help=cli_mlcp_help.get('start', 'start mlcp'))
def start():
    is_config = os.path.exists(os.path.join(os.getcwd(), 'config.py'))
    is_yaml = os.path.exists(os.path.join(os.getcwd(), 'deployment/docker-compose.yaml'))
    is_env = os.path.exists(os.path.join(os.getcwd(), 'env/.env'))

    if is_config and is_env and is_yaml:

        # find available port
        # init_ui_port = 8081
        # ui_port = get_available_port(init_ui_port)
        #
        # if ui_port != init_ui_port:
        #     click.echo('Warning: port ' + str(init_ui_port) +' already in use, using ' + str(ui_port) + ' instead.')

        # set env file
        set_env(env_filename='.env')

        # start command
        cmd = "docker-compose up"

        # set cwd to deployment folder
        os.chdir(os.path.join(os.getcwd(), "deployment"))

        os.system(cmd)

        # set cwd back to root
        os.chdir("../")
    else:
        click.secho("ERROR: Please run 'mlapp mlcp setup' command before starting.", fg='red')


@commands.command("stop", help=cli_mlcp_help.get('stop', 'stop mlcp'))
def stop():
    # set env file
    set_env(env_filename='')

    # set cwd to deployment folder
    os.chdir(os.path.join(os.getcwd(), "deployment"))

    cmd = "docker-compose down"
    os.system(cmd)

    # set cwd back to root
    os.chdir("../")
