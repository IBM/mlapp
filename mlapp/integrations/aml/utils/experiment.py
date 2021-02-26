from azureml.core.runconfig import RunConfiguration
from azureml.core import ScriptRunConfig
import os
from azureml.core import Experiment


def run_local_compute_experiment(ws, experiment_name, entry_script, source_directory=os.getcwd()):
    # Edit a run configuration property on the fly.
    run_local = RunConfiguration()
    run_local.environment.python.user_managed_dependencies = True

    exp = Experiment(workspace=ws, name=experiment_name)

    src = ScriptRunConfig(source_directory=source_directory, script=entry_script, run_config=run_local)
    run = exp.submit(src)
    run.wait_for_completion(show_output=True)
