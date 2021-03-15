import os
import json

# fix for mac osx
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

import click
from shutil import copyfile
from mlapp.mlapp_cli.common.cli_utilities import create_directory, create_file
from mlapp.mlapp_cli.common.files import amlignore_file, azureml_env_file, default_config_file
from mlapp.utils.general import get_project_root
from mlapp.integrations.aml.cli_help import cli_aml_help
from mlapp.integrations.aml.utils.workspace import init_workspace
from mlapp.integrations.aml.utils.datastore import get_datastore
from mlapp.integrations.aml.scripts.deploy_model import run_script as run_deploy_model_script
from mlapp.integrations.aml.scripts.publish_pipeline_endpoint import run_script as publish_pipeline_endpoint_script
from mlapp.integrations.aml.scripts.publish_multisteps_pipeline import run_script as \
    publish_multisteps_pipeline_script
from mlapp.integrations.aml.utils.cli_steps import steps
from mlapp.mlapp_cli.common.cli_utilities import clean_spaces
from azureml.core import Workspace

# constants
EMPTY_INPUT = ''

init_files_aml = {
    'aml_deployment.py': {
        'dir': 'deployment'
    },
    'aml_target_compute.py': {
        'dir': 'deployment'
    },
    'aml_flow.py': {
        'dir': 'deployment'
    }
}


@click.group("aml")
def commands():
    """
    MLApp AML Command

    Use it to run Azure Machine Learning commands.
    type --help on aml command to get more information.

    """
    pass


@commands.command("setup", help=cli_aml_help.get('setup'))
@click.option("-f", "--force", is_flag=True, default=False,
              help="Flag force setup if some of the AML files already exists in your project.")
def setup(force):
    if not force:
        is_initiated = False
        for file_name in init_files_aml:
            file_options = init_files_aml.get(file_name, file_name)
            directory = file_options.get('dir', 'root')
            if directory == 'root':
                full_path = os.path.join(os.getcwd(), file_name)
            else:
                if os.path.isdir(directory):
                    full_path = os.path.join(os.getcwd(), directory, file_name)
                else:
                    continue
            if os.path.exists(full_path):
                is_initiated = True
                break

        if is_initiated:
            click.secho(
                "ERROR: " + file_name + " already exists.\nHint: you can use 'mlapp aml setup --force' option to force setup (caution: force may override exsiting files).",
                fg='red')
            return

        # setup azure machine learning deployment files
        _setup_aml()
    else:
        _setup_aml()


def _setup_aml():
    # creates the deployment directory if not exists.
    create_directory(directory_name='deployment', include_init=False)

    # creates the environment directory if not exists.
    create_directory(directory_name='env', include_init=False)
    create_file(file_name='.env', path='env', content=azureml_env_file)

    # sets config.py environment file
    default_env_filename = ''
    config_file_content = default_config_file.replace("<FILENAME>", default_env_filename)
    create_file(file_name='config.py', content=config_file_content)

    for file_name in init_files_aml:
        file_options = init_files_aml.get(file_name, file_name)
        directory = file_options.get('dir', 'root')
        src = os.path.join(get_project_root(), 'mlapp', 'integrations', 'aml', 'generated_files', file_name)
        if directory == 'root':
            dst = os.path.join(os.getcwd(), file_name)
        else:
            dst = os.path.join(os.getcwd(), directory, file_name)

        # copy file from src to dst
        copyfile(src, dst)

    # creates amlignore
    create_file('.amlignore', content=amlignore_file)


@commands.command("deploy-model", help=cli_aml_help.get('deploy_model'))
@click.argument("experiment-name", required=True)
@click.argument("asset-name", required=True)
@click.option('-as', '--asset-label', default=None, help="Use it to add a label to your asset.")
@click.option('-rid', '--run-id', default=None, help="Use it to deploy a specific model.")
@click.option('-smetric', '--score-metric', default=None,
              help="Use it to choose best model according to a score metric (must be passed together with grater-is-better option).")
@click.option('-g', '--greater-is-better', default=False,
              help="Use it to set your score metric options (must be passed together with score-metric option).",
              type=bool)
@click.option('-cpu', '--cpu-cores', default=1, help="Use it to set number of cores in compute target machine.",
              type=int)
@click.option('-mgb', '--memory-gb', default=8, help="Use it to set memory size in compute target machine.", type=int)
def deploy_model(experiment_name, asset_name, asset_label, run_id, score_metric, greater_is_better, cpu_cores,
                 memory_gb):
    try:
        ws, _ = _get_aml_objects()
        run_deploy_model_script(ws, experiment_name, asset_name, asset_label, run_id, score_metric,
                                greater_is_better, cpu_cores, memory_gb)
    except Exception as e:
        click.secho(str(e), fg='red')


@commands.command("publish-pipeline", help=cli_aml_help.get('publish_pipeline'))
@click.argument("pipeline-name", required=True)
@click.argument("compute-target", required=True)
@click.option('-vs', '--vm-size', default='STANDARD_D2_V2', help="Use it to set vm size.")
@click.option('-mnn', '--min-nodes', default=0, help="Use it set min nodes number.", type=int)
@click.option('-mxn', '--max-nodes', default=4, help="Use it set max nodes number.", type=int)
def publish_pipeline(pipeline_name, compute_target, vm_size, min_nodes, max_nodes):
    try:
        ws, datastore = _get_aml_objects()
        publish_pipeline_endpoint_script(ws, datastore, pipeline_name, compute_target, vm_size, min_nodes,
                                         max_nodes)
    except Exception as e:
        click.secho(str(e), fg='red')


@commands.command("publish-multisteps-pipeline", help=cli_aml_help.get('publish_pipeline'))
@click.argument("pipeline-name", required=True)
def publish_multisteps_pipeline(pipeline_name):
    try:
        ws, datastore = _get_aml_objects()
    except:
        click.secho("ERROR: Please run 'mlapp init -aml' first or check your azureml Workspace credentials in config.py", fg='red')
        return

    number_of_steps_body = steps.get('number_of_steps', {})
    dependencies = number_of_steps_body.get('nested_dependency', {})
    number_of_steps = _display_message(number_of_steps_body)
    new_compute_targets = {}
    is_new_compue_target = False
    instructions = []
    for i in range(number_of_steps):
        args = {}
        is_default_values = False  # if true takes all default values for the test of the parameters.
        print('Please enter step %s arguments:' % str(i + 1))
        for dependency_name, body in dependencies.items():
            if is_default_values:
                args[dependency_name] = body.get('default', None)
            else:
                args[dependency_name] = _display_message(body)

            if dependency_name == 'compute_target':
                if args[dependency_name] in ws.compute_targets.keys():
                    break
                elif args[dependency_name] in new_compute_targets.keys():
                    args = new_compute_targets[args[dependency_name]]
                    break
                else:
                    click.echo('￿Found a new compute-target')
                    is_new_compue_target = True
            elif dependency_name == 'set_advanced' and not args[dependency_name]:
                # break if user don't want to set advanced settings
                is_default_values = True

        if is_new_compue_target:
            new_compute_targets[args['compute_target']] = args
            is_new_compue_target = False
        instructions.append(args)

    publish_multisteps_pipeline_script(ws, datastore, pipeline_name, instructions)


def _update_env_name(env_name):
    try:
        with open(os.path.join(os.getcwd(), 'config.py'), 'r') as f:
            config_content = f.read()
            exec(config_content)
            d = eval("settings")

        if 'aml' not in d:
            d['aml'] = {}

        d['aml']['environment'] = env_name

        config_file_content = '''settings = ''' + json.dumps(d, indent=2)
        create_file(file_name='config.py', content=config_file_content)

    except Exception as e:
        raise e


def _get_aml_objects():
    try:
        with open(os.path.join(os.getcwd(), 'config.py'), 'r') as f:
            config_content = f.read()
            exec(config_content)
            d: dict = eval("settings")

        # check if Workspace credentials exists
        aml = d.get('aml', {})
        if not aml:
            raise Exception("ERROR: please add AML Workspace credentials in your config.py under aml parameter.")

        tenant_id = aml.get('tenant_id')
        subscription_id = aml.get('subscription_id')
        resource_group = aml.get('resource_group')
        workspace_name = aml.get('workspace_name')
        datastore_name = aml.get('datastore_name', 'workspaceblobstore')

        if subscription_id is not None and resource_group is not None and workspace_name is not None:
            ws: Workspace = init_workspace(tenant_id, subscription_id, resource_group, workspace_name)
            datastore = get_datastore(ws, datastore_name)
            return ws, datastore
        else:
            raise Exception(
                "ERROR: credentials must include properties: subscription_id, resource_group, workspace_name.")
    except Exception as e:
        raise e


def _display_message(body):
    while True:
        display_name = body.get('display_name', '')
        short_description = body.get('short_description', '')

        # create message to display on the terminal.
        message = display_name + " (" + short_description + "): " if short_description != '' else display_name + ": "

        # get user input
        user_input = clean_spaces(input(message))

        # check for default and required values
        default_value = body.get('default', None)
        is_required = body.get('required', False)

        try:
            if default_value is not None and user_input == EMPTY_INPUT:
                user_input = default_value
            elif user_input == EMPTY_INPUT and is_required:
                raise Exception("'" + display_name + "' is required, please enter a valid value.")
            else:
                # validates user input
                validations_methods = body.get('validations', [])
                ans = True
                for validate in validations_methods:
                    ans &= validate(user_input)
                    if not ans:
                        raise Exception(body.get('error_msg', ''))

                # transforms user input
                transformations_methods = body.get('transformations', [])
                for trans in transformations_methods:
                    user_input = trans(user_input)

                # check for possible values
                body_values = body.get('values', None)
                if body_values:
                    body_values_keys = body['values'].keys()
                    if user_input in body_values_keys:
                        user_input = body['values'][user_input]
                    else:
                        raise Exception(body.get('error_msg', 'Oops something bad happened.'))
            break
        except Exception as e:
            click.secho(str(e), fg='red')

    return user_input
