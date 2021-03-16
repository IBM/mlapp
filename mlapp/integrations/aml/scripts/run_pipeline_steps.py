from mlapp.integrations.aml.utils.compute import get_or_create_compute_target
from mlapp.integrations.aml.utils.pipeline import create_mlapp_pipeline_step, run_pipeline_steps
from mlapp.integrations.aml.utils.runconfig import create_runconfig
import os


def run_script(ws, datastore, compute_target, vm_size='STANDARD_D2_V2', min_nodes=0, max_nodes=4):
    # init components
    compute_target = get_or_create_compute_target(ws, compute_target, vm_size, min_nodes, max_nodes)
    run_config = create_runconfig(compute_target)

    # create pipelines steps
    steps = create_mlapp_pipeline_step(
        compute_target, run_config,
        source_directory=os.getcwd(),
        entry_script=os.path.join("deployment", "aml_target_compute_run.py"))

    # publish pipeline endpoint
    run_pipeline_steps(ws, steps, experiment_name='pipeline_from_local')
