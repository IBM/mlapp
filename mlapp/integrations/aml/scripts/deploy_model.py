import os
from mlapp.integrations.aml.utils.deploy import get_best_model_in_experiment, deploy_model


def run_script(ws, experiment_name, asset_name, asset_label=None, run_id=None,
               score_metric=None, greater_is_better=False, cpu_cores=1, memory_gb=8):

    if run_id is None:
        # get best model in experiment
        run_id = get_best_model_in_experiment(
            ws, experiment_name, asset_name, asset_label, score_metric, greater_is_better)

    # deployment name
    if asset_label is not None:
        aci_service_name = asset_name.replace('_', '-') + '-' + asset_label.replace('_', '-')
    else:
        aci_service_name = asset_name.replace('_', '-')

    # entry script location
    entry_script = os.path.join("deployment", "aml_deployment.py")

    # deploy
    deploy_model(ws, aci_service_name, experiment_name, asset_name, asset_label, run_id, cpu_cores, memory_gb, entry_script)

