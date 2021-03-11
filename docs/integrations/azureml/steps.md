# Multi-Step Pipeline

### 1. Install MLApp
Use **aml** extras which will also install the azure machine learning packages. 
```
pip install "mlapp[aml]"
```

### 2. Run in terminal
```
mlapp aml setup
```

>Note: this will create 3 files: 
>
> `.amlignore` - This file will handle files to ignore when uploading snapshots to AzureML.
>
> `deployment/aml_target_compute.py` - This file will be the running script file when running ML App on AzureML compute.
>
> `deployment/aml_deployment.py` - This file will be the running script file when running ML App on AzureML deployment.
>
> `deployment/aml_flow.py` - This file will be the running script file when running Pipeline Steps on AzureML deployment.


### 3. Add into your `config.py` the AML subscription details:
```
"aml": {
    "tenant_id": "<tenant_id>>"
    "subscription_id": "<subscription_id>"
    "resource_group": "<resource_group>"
    "workspace_name": "<workspace_name>"
    "datastore_name": "<datastore_name>"
    "environment": "<environment_name>"
}
```

> Note: Use `mlapp/integrations/aml/create_mlapp_env.py` to create an environment with ML App installed in your AzureML workspace. 

### 4. Publish Multi Steps Pipeline Endpoint
```
mlapp aml publish-multisteps-pipeline <pipeline_endpoint_name>
```

> Note: you will be opted for how many steps you'd like to use, and each step you can choose which compute target to use.

Example of a 3 step pipeline:
![azureml-flow-steps](/integrations/azureml/imgs/azureml-flow-steps.png)


### 6. Use Pipeline Endpoint in the AzureML UI: [https://ml.azure.com/](https://ml.azure.com/)

- Go to "Endpoints" in the left navbar -> select "Pipeline endpoints" tab.
- Go to <pipeline_endpoint_name>
- Click on “Submit”.
- Create an experiment name.
> Note: we recommend using the asset name and/or asset label as the experiment name for easy identification
- Copy and split your Flow Config to the different steps you created in the right order: `config0`, `config1`, etc.
> Note: for a full example of how this is done check this example: [Running Flow in AzureML](/integrations/azureml/running-flow-in-azureml/).
- Hit the "Submit" button.
- Go to “Experiments” on the left navbar. Select your experiment.
- See results  - you will see your run in there as “running”
> Note: we highly recommend to customize your UI table:
> - Add filter: RunType “not in” azureml.PipelineRun
>
> - Toggle "Include child runs" - set to true.
>
> - Edit Table step 1: remove all columns and keep only: Run number, Status, Duration. 
>
> - Edit Table step 2: add the following columns: asset_name, asset_label, pipeline, run_id and any metric scores you have.
>
> - Edit Table setp 3: Edit the figures to use one of your score metrics.
