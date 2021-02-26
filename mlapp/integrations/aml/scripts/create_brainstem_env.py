import subprocess
import sys
from mlapp.utils.generic_utils import get_project_root
from mlapp.version import VERSION
import os

os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

from mlapp.integrations.aml.utils.env import create_or_update_mlapp_env, \
    display_mlapp_environments
from mlapp.integrations.aml.utils.workspace import init_workspace

try:
    from mlapp.integrations.aml.scripts.workspace_config import workspace_config
except:
    workspace_config = None

'''
*** INSTRUCTIONS FOR USING THIS SCRIPT ***

1. Update ML App VERSION.
2. Rebuild mlapp wheel file: `python3 setup.py bdist_wheel`.
3. Open `requirements.txt` file:
    3.1. Unmark `azureml-sdk==1.0.83` and `azureml-defaults==1.0.83`.
    3.2. Add any other custom libraries you want in your environment.
4. Copy your workspace's config.json to workspace_config.py in `mlapp/integrations/aml/scripts/workspace_config.py`.
5. Run script.
6. Revert `requirements.txt` so the changes don't get pushed.
'''


def _run_from_mlapp_main_library():
    wheel_path_ending = '-py3-none-any.whl'

    # authenticate into workspace
    if workspace_config is None:
        raise Exception("Missing workspace's config.json in "
                        "`mlapp/integrations/aml/scripts/workspace_config.py`")

    workspace = init_workspace(
        workspace_config['tenant_id'],
        workspace_config['subscription_id'],
        workspace_config['resource_group'],
        workspace_config['workspace_name']
    )

    # set current working directory to root project
    os.chdir(get_project_root())

    # use current library's version
    version_number = str(VERSION)

    return workspace, wheel_path_ending, version_number


def _run_from_child_library(git_url, version_number):
    wheel_path_ending = '.zip'

    # set up the version of mlapp to use
    if version_number is None:
        version_number = str(VERSION)

    # extension for git url
    version_extension = '@' + version_number

    if git_url is None:
        git_url = "git+ssh://git@github.com/ibm/mlapp.git"

    os.makedirs('dist', exist_ok=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "download", '{0}{1}'.format(git_url, version_extension),
         "--no-deps", "--dest", "dist/"])

    return wheel_path_ending, version_number


def _handle_requirments(requirements):
    created_requirments_file = False

    # create requirements file based on installed libraries
    if requirements is None:
        requirements = "dist/requirements.txt"
        os.makedirs('dist', exist_ok=True)
        with open(requirements, "w") as f:
            subprocess.check_call([sys.executable, "-m", "pip", "freeze", ">", requirements], stdout=f)

        # remove mlapp from requirements
        with open(requirements, "r") as f:
            lines = f.readlines()
        with open(requirements, "w") as f:
            for line in lines:
                if not line.strip("\n").startswith("mlapp"):
                    f.write(line)

        created_requirments_file = True

    # use requirements file provided
    else:
        if not os.path.exists(requirements):
            raise Exception("ERROR: requirements file path not found.")

    return requirements, created_requirments_file


def run_script(workspace, git_url, version_number, requirements):
    if workspace is None:
        # running from mlapp main library
        workspace, wheel_path_ending, version_number = _run_from_mlapp_main_library()
    else:
        # running from mlapp "child" library
        if version_number is None:
            version_number = str(VERSION)

        if 'v' not in version_number:
            version_number = 'v{0}'.format(version_number)
        try:
            wheel_path_ending, version_number = _run_from_child_library(git_url, version_number)
        except:
            wheel_path_ending, version_number = _run_from_child_library(git_url, version_number.replace('v', ''))

    # handle requirements
    requirements, created_requirments_file = _handle_requirments(requirements)

    # set name for new environment in AzureML
    env_name = "mlapp-" + version_number

    # create/update environment
    wheel_path = 'dist/mlapp-{0}{1}'.format(version_number.replace('v', ''), wheel_path_ending)
    create_or_update_mlapp_env(workspace, requirements_path=requirements, wheel_path=wheel_path, env_name=env_name)

    # cleanup
    os.remove(wheel_path)
    if created_requirments_file:
        os.remove(requirements)
    if not os.listdir('dist'):
        os.rmdir('dist')

    return env_name

# environment_name = run_script(None, None, None, 'requirements.txt')
# print(environment_name)

# print_mlapp_environments = False
# if print_mlapp_environments:
#     display_mlapp_environments(workspace)


