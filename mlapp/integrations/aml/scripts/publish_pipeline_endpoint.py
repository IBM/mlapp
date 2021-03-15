from mlapp.integrations.aml.utils.compute import get_or_create_compute_target
from mlapp.integrations.aml.utils.pipeline import publish_pipeline_endpoint, create_mlapp_pipeline_step
from mlapp.integrations.aml.utils.runconfig import create_runconfig
import os


def run_script(ws, datastore, pipeline_name, compute_target, vm_size='STANDARD_D2_V2', min_nodes=0, max_nodes=4):
    # init components
    compute_target = get_or_create_compute_target(ws, compute_target, vm_size, min_nodes, max_nodes)
    run_config = create_runconfig(compute_target)

    # create pipelines steps
    steps = create_mlapp_pipeline_step(
        compute_target, run_config,
        source_directory=os.getcwd(),
        entry_script=os.path.join("deployment", "aml_target_compute.py"))

    # publish pipeline endpoint
    publish_pipeline_endpoint(ws, steps, pipeline_name)


