import os
import click
from shutil import copyfile
from mlapp.integrations.sm.cli_help import cli_sm_help
from mlapp.mlapp_cli.common.cli_utilities import create_directory, create_file
from mlapp.mlapp_cli.common.files import dockerignore_file, default_config_file
from mlapp.utils.general import get_project_root
from mlapp.integrations.sm.scripts.run_training_job import run_training_job as run_training_job_script
from mlapp.integrations.sm.scripts.run_tuning_job import run_tuning_job as run_tuning_job_script


init_files_sm = {
    'sm_deployment.py': {
        'dir': 'deployment'
    }
}


@click.group("sm")
def commands():
    """
    MLApp SM Command

    Use it to run Sage Maker commands.
    type --help on sm command to get more information.

    """
    pass


@commands.command("setup", help=cli_sm_help.get('setup'))
@click.option("-f", "--force", is_flag=True, default=False,
              help="Flag force setup if some of the SageMaker files already exists in your project.")
def setup(force):
    if not force:
        is_initiated = False
        for file_name in init_files_sm:
            file_options = init_files_sm.get(file_name, file_name)
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
                "ERROR: " + file_name + " already exists.\nHint: you can use 'mlapp sm setup --force' option to force setup (caution: force may override exsiting files).",
                fg='red')
            return

        # setup azure machine learning deployment files
        _setup_sm()
    else:
        _setup_sm()


def _setup_sm(skip_dockerignore=False):
    # creates the deployment directory if not exists.
    create_directory(directory_name='deployment', include_init=False)

    # creates the environment directory if not exists.
    create_directory(directory_name='env', include_init=False)
    create_file(file_name='.env', path='env', content=default_config_file)

    # sets config.py environment file
    default_env_filename = ''
    config_file_content = default_config_file.replace("<FILENAME>", default_env_filename)
    create_file(file_name='config.py', content=config_file_content)

    for file_name in init_files_sm:
        file_options = init_files_sm.get(file_name, file_name)
        directory = file_options.get('dir', 'root')
        src = os.path.join(get_project_root(), 'mlapp', 'integrations', 'sm', 'generated_files', file_name)
        if directory == 'root':
            dst = os.path.join(os.getcwd(), file_name)
        else:
            dst = os.path.join(os.getcwd(), directory, file_name)

        # copy file from src to dst
        copyfile(src, dst)

    # creates dockerignore
    if not skip_dockerignore:
        if not os.path.exists(os.path.join(os.getcwd(), '.dockerignore')):
            create_file('.dockerignore', content=dockerignore_file)


@commands.command("run-training-job", help=cli_sm_help.get('run_training_job'))
@click.argument("base-job-name", required=True)
@click.argument("image-name", required=True)
@click.argument("config", required=True)
@click.argument("metrics", required=True)
@click.option('-it', '--instance-type', default='ml.m4.xlarge', help="Default value is `ml.m4.xlarge`.")
@click.option('-ic', '--instance-count', default=1, help="Number of instances. Default is 1.", type=int)
def run_training_job(base_job_name, image_name, config, metrics, instance_type, instance_count):
    try:
        _ = _get_sm_objects()
        run_training_job_script(base_job_name, image_name, config, metrics, instance_type, instance_count)
    except Exception as e:
        click.secho(str(e), fg='red')


@commands.command("run-tuning-job", help=cli_sm_help.get('run_tuning_job'))
@click.argument("base-job-name", required=True)
@click.argument("image-name", required=True)
@click.argument("config", required=True)
@click.argument("metrics", required=True)
@click.argument("objective-metric-name", required=True)
@click.argument("hyperparameter-ranges", required=True)
@click.option('-it', '--instance-type', default='ml.m4.xlarge', help="Default value is `ml.m4.xlarge`.")
@click.option('-ic', '--instance-count', default=1, help="Number of instances. Default is 1.", type=int)
@click.option('-mpj', '--max-parallel-jobs', default=1, help="Number of instances. Default is 3.", type=int)
@click.option('-mx', '--max-jobs', default=1, help="Number of instances. Default is 3.", type=int)
def run_tuning_job(base_job_name, image_name, config, metrics, objective_metric_name, hyperparameter_ranges,
                   instance_type, instance_count, max_parallel_jobs, max_jobs):
    try:
        _ = _get_sm_objects()
        run_tuning_job_script(base_job_name, image_name, config, metrics, objective_metric_name, hyperparameter_ranges,
                              instance_type, instance_count, max_parallel_jobs, max_jobs)
    except Exception as e:
        click.secho(str(e), fg='red')


def _get_sm_objects():
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
