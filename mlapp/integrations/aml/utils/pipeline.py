from azureml.core import Experiment, Workspace
from azureml.pipeline.core import PipelineParameter, Pipeline, PublishedPipeline
from azureml.pipeline.steps import PythonScriptStep, DatabricksStep

from mlapp.integrations.aml.utils.constants import INPUT_DATA_ARGUMENT, OUTPUT_DATA_ARGUMENT


def create_mlapp_pipeline_step(compute_target, run_config, source_directory, entry_script,
                                   input_dir=None, output_dir=None, param_name='config'):
    # adding pipeline parameter
    pipeline_param = PipelineParameter(
        name=param_name,
        default_value='{"pipelines_configs": [{"data_settings": {}, "model_settings": {}, '
                      '"job_settings":{"pipeline": "", "asset_name": ""}}]}')

    arguments = [("--" + param_name), pipeline_param]
    if input_dir is not None or output_dir is not None:
        if input_dir is not None and not isinstance(input_dir, list):
            input_dir = [input_dir]

        if output_dir is not None and not isinstance(output_dir, list):
            output_dir = [output_dir]

        for dir_type in [{'value': input_dir, 'str': INPUT_DATA_ARGUMENT},
                         {'value': output_dir, 'str': OUTPUT_DATA_ARGUMENT}]:
            if dir_type['value'] is not None:
                for i in range(len(dir_type['value'])):
                    arguments.append('--' + dir_type['str'] + str(i))
                    arguments.append(dir_type['value'][i])

    # pipeline step
    return [PythonScriptStep(
        script_name=entry_script,
        arguments=arguments,
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=False,
        inputs=input_dir, outputs=output_dir,
        source_directory=source_directory
    )]


def create_databricks_pipeline_step(compute_target, source_directory, entry_script, num_workers=1):
    pipeline_param = PipelineParameter(
        name="config",
        default_value='{"pipelines_configs": [{"data_settings": {}, "model_settings": {}, '
                      '"job_settings":{"pipeline": "", "asset_name": ""}}]}')

    # input directory in datastore
    # input_dir = DataReference(datastore=datastore, data_reference_name="input_dir", path_on_datastore="configs",
    #     mode='download')

    # output directory in datastore
    # output_dir = PipelineData(name="output_dir", datastore=datastore, output_path_on_compute='results')

    dbStep = DatabricksStep(
        'databricksstep',
        inputs=[pipeline_param],    # ,input_dir],
        # outputs=[output_dir],
        num_workers=num_workers,
        source_directory=source_directory,
        python_script_path=entry_script,
        compute_target=compute_target,
        allow_reuse=False
    )
    return [dbStep]


def publish_pipeline_endpoint(workspace, steps, name='mlapp', description='My Published ML App', version='1.0'):
    # publish pipeline
    mlapp_pipeline = Pipeline(workspace=workspace, steps=steps)
    mlapp_pipeline.publish(name=name, description=description, version=version)


def run_pipeline_endpoint(workspace: Workspace, pipeline_endpoint_id, experiment_name, config_str,
                          pipeline_version=None):
    pipeline_endpoint_by_name = PublishedPipeline.get(workspace=workspace, id=pipeline_endpoint_id)

    run_id = pipeline_endpoint_by_name.submit(
        workspace=workspace,
        experiment_name=experiment_name,
        pipeline_parameters={
            'config': config_str
        })
    print(run_id)

    # option 2 with active directory
    # aad_token = InteractiveLoginAuthentication(tenant_id="my-tenant-id")
    # rest_endpoint = ''
    # response = requests.post(rest_endpoint, headers=aad_token, json={"ExperimentName": "PipelineEndpointExperiment",
    #                          "RunSource": "API", "ParameterAssignments": config_str})


def run_pipeline_steps(workspace, steps, experiment_name):
    mlapp_pipeline = Pipeline(workspace=workspace, steps=steps)
    pipeline_run = Experiment(workspace, experiment_name).submit(mlapp_pipeline)
    pipeline_run.wait_for_completion()
