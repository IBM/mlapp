import os
from mlapp.config import settings
from mlapp.integrations.aml.utils.experiment import run_local_compute_experiment
from mlapp.integrations.aml.utils.workspace import init_workspace

# init workspace
ws = init_workspace(
    settings['aml']['tenant_id'],
    settings['aml']['subscription_id'],
    settings['aml']['resource_group'],
    settings['aml']['workspace_name']
)

run_local_compute_experiment(ws, experiment_name='local_experiment', entry_script=os.path.join("deployment", 'aml_target_compute.py'))
