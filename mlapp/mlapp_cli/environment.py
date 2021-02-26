import click, os
from mlapp.mlapp_cli.common.cli_utilities import set_env, create_file, create_directory
from mlapp.mlapp_cli.cli_help import cli_environment_help


@click.group("environment")
def commands():
    """
    ML App Environment Command

    """
    pass


@commands.command("init", help=cli_environment_help.get('init', 'init environment file'))
@click.argument("name", required=True, default='.env')
def init(name):
    try:
        if '.env' not in name:
            name += '.env'

        env_full_path = os.path.join(os.getcwd(), os.path.join('env', name))
        if not os.path.exists(env_full_path):

            # creates the env directory if not exists.
            create_directory(directory_name='env', include_init=False)

            # creates env file
            create_file(name, path='env')

            # set the new env file
            set_env(name)
        else:
            click.secho("ERROR: '" + name + "' file already exits.", fg='red')

    except Exception as e:
        click.secho("ERROR: Oops, something went wrong.", fg='red')


@commands.command("set", help=cli_environment_help.get('set', 'sets environment file'))
@click.argument("name", required=True)
def set(name):
    try:
        if '.env' not in name:
            name += '.env'

        # set the new env file
        set_env(name)

    except Exception as e:
        click.secho("ERROR: Oops, something went wrong.", fg='red')
