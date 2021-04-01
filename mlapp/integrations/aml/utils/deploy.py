import os
import shutil

from mlapp.integrations.aml.utils.env import create_env_from_requirements
from mlapp.utils.general import create_directory, create_tempdir, delete_directory_with_all_contents
from azureml.core import Webservice, Run, Experiment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException
from mlapp import MLApp
from mlapp.config import settings
from mlapp.managers.flow_manager import FlowManager
from mlapp.integrations.aml.utils.run_class import get_model_register_name
from mlapp.integrations.aml.utils.constants import AML_MLAPP_FOLDER


def deploy_model(
        ws, aci_service_name, experiment_name, asset_name, asset_label, run_id, cpu_cores, memory_gb, entry_script):
    env = create_env_from_requirements(endpoint=True)
    inference_config = InferenceConfig(source_directory=os.getcwd(),
                                       entry_script=entry_script,
                                       environment=env)

    deployment_config = AciWebservice.deploy_configuration(cpu_cores=cpu_cores, memory_gb=memory_gb)

    # model name
    model_name = get_model_register_name(run_id)
    try:
        model = Model(ws, name=model_name)
    except:
        # creating directory for download Model files for Model register
        tmp_path = create_tempdir(name='download_tmp')
        register_path = create_directory(AML_MLAPP_FOLDER, path=tmp_path)

        # getting RUN context
        experiment = Experiment(workspace=ws, name=experiment_name)
        tags = {"run_id": run_id, "asset_name": asset_name}
        if asset_label is not None:
            tags["asset_label"] = asset_label

        selected_run_id = None
        for run in Run.list(experiment, tags=tags, include_children=True, status='Completed'):
            run_metrics = run.get_metrics()
            exp_saved_run_id = run_metrics.get("run_id")
            if exp_saved_run_id == run_id:
                selected_run_id = run.id
                break
        if selected_run_id is None:
            raise Exception('ERROR: there is no matching Run object that associated with the run id %s in this experiment.' % str(run_id))
        current_run = Run(experiment=experiment, run_id=selected_run_id)

        # download files from run object
        current_run.download_files(output_directory=register_path)

        # register model
        model = Model.register(
            ws,
            model_path=register_path,
            model_name=model_name,
            tags=tags,
            description=asset_name)

        # deletes tmp dir and all content
        delete_directory_with_all_contents(tmp_path)

    # deploy model
    service = None
    try:
        service = Webservice(ws, name=aci_service_name)
        service.update(models=[model], inference_config=inference_config)
    except WebserviceException as e:
        if service:
            service.delete()
        service = Model.deploy(ws, aci_service_name, [model], inference_config, deployment_config)

    service.wait_for_deployment(True)


def get_best_model_in_experiment(ws, experiment_name, asset_name, asset_label, score_metric, greater_is_better):
    best_score_run_id = None
    best_score = None

    tags = {
        'asset_name': asset_name
    }
    if asset_label is not None:
        tags['asset_label'] = asset_label

    # for run in experiment.get_runs(tags=tags, include_children=True):
    experiment = ws.experiments[experiment_name]
    for run in Run.list(experiment, tags=tags, include_children=True, status='Completed'):
        run_metrics = run.get_metrics()
        run_details = run.get_details()
        # each logged metric becomes a key in this returned dict
        run_score = run_metrics.get(score_metric)
        if run_score is not None:
            run_id = run_metrics["run_id"]

            if best_score is None:
                best_score = run_score
                best_score_run_id = run_id
            else:
                if greater_is_better:
                    if run_score > best_score:
                        best_score = run_score
                        best_score_run_id = run_id
                else:
                    if run_score < best_score:
                        best_score = run_score
                        best_score_run_id = run_id

    if not best_score:
        raise Exception(f"Error: score metric '{score_metric}' was not found in any run!")

    if not best_score_run_id:
        raise Exception(f"Error: haven't found a run with score metric '{score_metric}' score metric.")

    print("Best model run_id: " + best_score_run_id)
    print("Best model score: " + str(best_score))

    return best_score_run_id


def insert_model_id(configuration, model_id):
    job_settings = configuration['pipelines_configs'][0]['job_settings']
    if 'model_id' not in job_settings:
        job_settings['model_id'] = model_id


def preprocess_deployment(model_path):
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    run_id = None

    mlapp_files_path = os.path.join(model_path, 'mlapp')

    for file_name in os.listdir(mlapp_files_path):
        # TODO: fetch this from logger txt file
        if '.pkl' in file_name:
            run_id = str(file_name.split('.')[0]).split('_')[0]

        shutil.move(os.path.join(mlapp_files_path, file_name), output_dir)

    return run_id


def get_predictions_path(run_ids):
    for run_id in run_ids:
        for file_name in os.listdir('output'):
            if 'predictions' in file_name and run_id in file_name:
                return os.path.join('output', file_name)

