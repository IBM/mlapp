# Getting Started with AzureML

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


### 3. Add into your `config.py` the AML subscription details:
```
"aml": {
    "subscription_id": "<subscription_id>",
    "resource_group": "<resource_group>",
    "workspace_name": "<workspace_name>",
    "datastore_name": "<datastore_name>(Optional)",
    "tenant_id": "<tenant_id>(Optional)"
}
```

!!! tip "AzureML Configuration"

    **tenant_id** can be retrieved in the azure portal through navigation to the active directory. It might be **required** depending on the settings of your active directory. 
    
    **datastore_name** is optional and the default workspace's datastore can be used, although it is **recommended** to have a separated blob storage and connect it to the workspace via **datastore** for storing the MLApp output files.  

### 4. Run in terminal:  
```
mlapp aml -h
``` 

> Note: will show you the available commands: 
>
> `publish-pipeline` - Will create a new pipeline endpoint with your current code.
>
> `deploy-model` - Deploy will create a webservice given a model_id or will promote the best model by itself.
>
> `publish-multisteps-pipeline` - Will create a new pipeline endpoint supporting multiple steps with your current code.

### 5. Publish pipeline endpoint
```
mlapp aml publish-pipeline <pipeline_endpoint_name> <compute_target_name> --vm-size <vm_size> --min-nodes <min_nodes> --max-nodes <max_nodes>
```

!!! note "Creating a Compute Cluster"
    
    You must create a compute cluster in order to publish a pipeline. You can do that via the AzureML workspace. Look for **Compute** in the navigation and create a compute under the **Compute clusters** tab  
    
### 6. Use pipeline endpoint in the AzureML UI: [https://ml.azure.com/](https://ml.azure.com/)

- Go to "Endpoints" in the left navbar -> select "Pipeline endpoints" tab.
- Go to <pipeline_endpoint_name>
- Click on “Submit”.
- Create an experiment name.
> Note: we recommend using the asset name and/or asset label as the experiment name for easy identification
- Copy your train config paste it in the config parameter.
- Hit the "Submit" button.
- Go to “Experiments” on the left navbar. Select your experiment.
- See results  - you will see your run in there as “running”
> Note: we highly recommend to customize your UI table:
>
> - Add filter: RunType “not in” azureml.PipelineRun
>
> - Toggle "Include child runs" - set to true.
>
> - Edit Table step 1: remove all columns and keep only: Run number, Status, Duration.
> 
> - Edit Table step 2: add the following columns: asset_name, asset_label, pipeline, run_id and any metric scores you have.
>
> - Edit Table setp 3: Edit the figures to use one of your score metrics.

Example view:
![azureml-experiment-view](/integrations/azureml/imgs/azureml-experiment-view.png)

### 7. Deploy a web service

There are two methods to deploy a service: 

**First** - you can promote the best model in the experiment by one of your metric scores.
```
mlapp aml deploy-model <asset_name>> -expn <experiment_name> -smetric <score_metric_name> -g <greater_is_better> --asset-label <asset_label>
``` 
 
**Second** - you can manually select a run id and deploy it.
```  
mlapp aml deploy-model <asset_name> -rid <run_id>  --asset-label <asset_label>
``` 

A new deployment should be spinning up for you to use! Check "Endpoints" under "Real-time endpoints".
