import click
import os

# mlapp CLI
import mlapp.mlapp_cli.assets
import mlapp.mlapp_cli.boilerplates
import mlapp.mlapp_cli.services
import mlapp.mlapp_cli.environment
import mlapp.mlapp_cli.mlcp
from mlapp.mlapp_cli.init import init_command
from mlapp.mlapp_cli.common.files import gitignore_file, dockerignore_file
from mlapp.mlapp_cli.common.cli_utilities import create_file
from mlapp.version import VERSION
try:
    from mlapp.mlapp_cli.cli_test_env import check_env_test
except ModuleNotFoundError:
    check_env_test = None

try:
    from mlapp.integrations.aml.cli import commands as aml_commands
except:
    aml_commands = None

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

if check_env_test and callable(check_env_test):
    check_env_test()

# ASCII - standard font
LOGO = r""" 
IBM Services Framework for
███    ███ ██       █████  ██████  ██████  
████  ████ ██      ██   ██ ██   ██ ██   ██ 
██ ████ ██ ██      ███████ ██████  ██████  
██  ██  ██ ██      ██   ██ ██      ██      
██      ██ ███████ ██   ██ ██      ██       {}
""".format(VERSION)


@click.group(name="mlapp", context_settings=CONTEXT_SETTINGS)
@click.version_option("", "--version", "-V", "-v", help="Show version and exit", prog_name="", message=LOGO)
def cli():
    pass


@cli.command(help="Creates an initial project structure")
@click.option("-mlcp", "--ml-control-panel", is_flag=True, default=False,
              help="Flag that includes the ML control panel in your project.")
@click.option("-aml", "--azure-machine-learning", is_flag=True, default=False, hidden=aml_commands is None,
              help="Flag that includes the AML setup files in your project.")
@click.option("-g", "--gitignore", is_flag=True, default=False,
              help="Flag that disables addition of a .gitignore file into your project.")
@click.option("-d", "--dockerignore", is_flag=True, default=False,
              help="Flag that disables addition of a .dockerignore file into your project.")
@click.option("-f", "--force", is_flag=True, default=False,
              help="Flag force init if project folder is not empty.")
def init(ml_control_panel, azure_machine_learning, gitignore, dockerignore, force):
    init_command(ml_control_panel, azure_machine_learning, gitignore, dockerignore, force)


@cli.command(help="Use to create ML App recommended '.gitigonre' file.")
@click.option("-f", "--force", is_flag=True, default=False,
              help="Flag force will override existing '.gitignore' file.")
def create_gitignore(force):
    if not os.path.exists(os.path.join(os.getcwd(), '.gitignore')) or force:
        create_file('.gitignore', content=gitignore_file)
    else:
        click.echo("Error: '.gitignore' already exists.")


@cli.command(help="Use to create ML App recommended '.dockerignore' file.")
@click.option("-f", "--force", is_flag=True, default=False,
              help="Flag force will override existing '.dockerignore' file.")
def create_dockerignore(force):
    if not os.path.exists(os.path.join(os.getcwd(), '.dockerignore')) or force:
        create_file('.dockerignore', content=dockerignore_file)
    else:
        click.echo("Error: '.dockerignore' already exists.")


cli.add_command(mlapp.mlapp_cli.assets.commands)
cli.add_command(mlapp.mlapp_cli.boilerplates.commands)
cli.add_command(mlapp.mlapp_cli.services.commands)
cli.add_command(mlapp.mlapp_cli.environment.commands)
cli.add_command(mlapp.mlapp_cli.mlcp.commands)

# adds integrations CLI's
if aml_commands is not None:
    cli.add_command(aml_commands)

if __name__ == "__main__":
    pass
