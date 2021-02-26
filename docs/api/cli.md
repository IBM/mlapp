# Command line interface (CLI)
Included with mlapp is a user friedly command line interface (CLI) which can be used to automate commonly used tasks. The CLI comes built with a through documentation which can be accesed by typing the following in the temrinal:  
​
```bash
mlapp --help
```
​
Should output the following:
```text   
    Usage: mlapp [OPTIONS] COMMAND [ARGS]...
    
    Options:
      -V, -v, --version  Show version and exit
      --help             Show this message and exit.
    
    Commands:
      assets               ML App Assets Command Use it to install or create...
      boilerplates         ML App Boilerplate's Command Use it to install...
      create-dockerignore  Use to create mlapp recommended '.dockerignore'...
      create-gitignore     Use to create mlapp recommended '.gitigonre'...
      environment          ML App Environment Command
      init                 Creates an initial project structure
      mlcp                 ML App MLCP Command Use it to setup and run...
      services             ML App Services Command
```  
		 
## 1. ML App Commands

### 1.1. assets 
The ```assets``` command is used to create your Assets. 
​
##### 1.1.1 Options:
* -h, --help &nbsp; Show this message and exit.
​
##### 1.1.2. Commands:
* **create:**	
​
> -- `mlapp assets create ASSET_NAME` - creates an asset named ASSET_NAME  
>   
> -- `mlapp assets create ASSET_NAME -f` - forcing asset creation named ASSET_NAME (caution: can override existing asset)
>
>>  create options:
>>
>> * -f, --force &nbsp; Flag force will override existing asset_name file
>>
>> * -w, --with-flow &nbsp; Creates asset train and forecast configs with flow settings
>>
>>> * -h, --help &nbsp; Show this message and exit.
​
​
* **rename:**  
​
> -- `mlapp assets rename PREVIOUS_NAME NEW_NAME` - duplicates existing asset named PREVIOUS_NAME and renamed it NEW_NAME  
> 
> -- `mlapp assets rename PREVIOUS_NAME NEW_NAME -d` - rename existing asset named PREVIOUS_NAME and renamed it NEW_NAME (deletes PREVIOUS_NAME asset)
>
>>  rename options:
>>
>> * -d, --delete &nbsp; Use it to delete previous asset directory on renaming.
>>
>>> * -h, --help &nbsp; Show this message and exit.
​
* **show:**    
​
> -- Show all your project assets.
>
>>  show options:
>> * -h, --help &nbsp; Show this message and exit.
​
---
​
### 1.2. boilerplates   
The ```boilerplate``` command is used to install [ML App Boilerplates](todo add link).  
​
##### 1.2.1. Options:
* -h, --help &nbsp; Show this message and exit.
​
##### 1.2.1. Commands:
* **install:**  
​
> -- `mlapp boilerplates install BOILERPLATE_NAME` - install mlapp boilerplate named BOILERPLATE_NAME  
> 
> -- `mlapp boilerplates install BOILERPLATE_NAME -f` - forcing boilerplate installation named BOILERPLATE_NAME (can override existing boilerplate)  
> 
> --`mlapp boilerplates install BOILERPLATE_NAME -r NEW_NAME` - install mlapp boilerplate named BOILERPLATE_NAME and rename it to NEW_NAME.
>
>
>>  install options:
>>
>> * -f, --force &nbsp; Flag force will override existing asset_name file.
>>
>> * -r, --new-name &nbsp; TEXT Use it to rename an asset name on installation.
>>
>> * -h, --help &nbsp; Show this message and exit.
​
* **show:**  
​
>Show all mlapp available boilerplate's.
>
>>  show options:
>>
>> * -h, --help &nbsp; Show this message and exit.
​
---
### 1.3. create-dockerignore 
Use to create mlapp recommended '.dockerignore' file.
​
##### 1.3.1. Options:
* -h, --help &nbsp; Show this message and exit.
* -f, --force &nbsp; force will override existing '.dockerignore' file.
​
---		 
### 1.4. create-gitignore  
Use to create mlapp recommended '.gitignore' file.
​
##### 1.4.1. Options:
* -h, --help &nbsp; Show this message and exit.
* -f, --force &nbsp; force will override existing '.gitignore' file.
​
---
### 1.5. environment 
MLApp Environment Command
	
##### 1.5.1. Options:
* -h, --help &nbsp; Show this message and exit.
​
##### 1.5.2. Commands:
* **init:** 
​
> -- `mlapp environment init` - creates an empty env file with default name '.env'.  
> -- `mlapp environment init NAME` - creates an empty env file named NAME.
​
* **set:**  
​
> `mlapp environment set NAME` - sets mlapp to point on NAME environment file, (this command modifies config.py file).
​
--- 
### 1.6. init  
​
Creates an initial project structure
		
##### 1.6.1. Options:
* -mlcp, --ml &nbsp; -control-panel  Flag that includes the ML control panel in your project.
* -g, --gitignore &nbsp; Flag that adds .gitignore file into your project.
* -d, --dockerignore &nbsp; Flag that adds .dockerignore file into your project.
* -f, --force &nbsp; Flag force init if project folder is not empty.
* --help &nbsp; Show this message and exit.
​
---
### 1.7. mlcp 
Use it to setup and run mlapp MLCP locally on your machine.
	
##### 1.7.1. Options:
* --help &nbsp; Show this message and exit.
	
##### 1.7.2. Commands:
* **setup:** 
​
> -- `mlapp mlcp setup` - setup mlcp on your machine.
>
>>  setup options:
>>
>> * -h, --help &nbsp; Show this message and exit.
 
* **start:** 
​
> --`mlapp mlcp start` - starts mlcp docker container, open chrome browser with the address [http://localhost:8081](http://localhost:8081).
>
>>  start options:
>>
>> * -h, --help &nbsp; Show this message and exit.  
​
* **stop:** 
​
> -- `mlapp mlcp stop` - stops mlcp docker container.
>
>>  stop options:
>> * -h, --help &nbsp; Show this message and exit.  
​
---
### 1.8. services  
Use it to setup and run mlapp MLCP locally on your machine.
	
##### 1.8.1. Options:
* --help &nbsp; Show this message and exit.
	
##### 1.8.2. Commands:
* **add:** 
​
> -- `mlapp services add SERVICE_NAME` - register a new service to mlapp (hint: use `mlapp services show-types`
to see all mlapp available services).
>
>>  add options:
>>
>> * -h, --help &nbsp; Show this message and exit.
​
* **delete:**  
​
> -- `mlapp services delete SERVICE_NAME` - deletes a registered service (hint: use `mlapp services show`
to see all your registered services).
>
>>  delete options:
>>
>> * -h, --help &nbsp; Show this message and exit.
​
* **show:** 
​
> -- `mlapp services show` - shows all your registered services.
>
>>  show options:
>>
>> * -h, --help &nbsp; Show this message and exit.
​
* **show_types**:
​
> -- `mlapp services show-types` - shows all mlapp available services.
>
>>  show-types options:
>>
>> * -h, --help &nbsp; Show this message and exit.

​
## 2. AzureML Commands
For Azure Machine Learning users, ML App also includes command line interface for aml deployment
In order to access aml commands `azureml-sdk` and `azureml-defaults` package should be installed in your project virtual environment.
​```bash
mlapp --help
​```
Should output the following:
​
```text
    Usage: mlapp [OPTIONS] COMMAND [ARGS]...
    
    Options:
      -V, -v, --version  Show version and exit
      -h, --help         Show this message and exit.
    
    Commands:
      aml                  ML App AML Command Use it to run Azure Machine...
      assets               ML App Assets Command Use it to install or create...
      boilerplates         ML App Boilerplate's Command Use it to install...
      create-dockerignore  Use to create ml app recommended '.dockerignore'...
      create-gitignore     Use to create ml app recommended '.gitigonre'...
      environment          ML App Environment Command
      init                 Creates an initial project structure
      mlcp                 ML App MLCP Command Use it to setup and run...
      services             ML App Services Command
```
    
 * NOTICE: `aml` command should appear under `Commands` section.
 
## 2.1. Commands
###setup 
The ```setup``` command is used to setup azureml configurations in your project. 
​
##### 2.1.1. Options:
*  -f, --force &nbsp; Flag force setup if some of the AML files already exists in your project.
* -h, --help &nbsp; Show this message and exit.
​
### 2.2. deploy-model 
The ```deploy-model``` command is used to deploy your model in azureml. 
​
##### 2.2.1. Arguments:
* ASSET_NAME &nbsp; enter your asset name to deploy
​
##### 2.2.2. Options:
* -as, --asset-label TEXT &nbsp; Use it to add a label to your asset.
* -rid, --run-id TEXT &nbsp; Use it to deploy a specific model.
* -smetric, --score-metric TEXT &nbsp; Use it to choose best model according to a score metric (must be passed together with grater-is-better option).
* -g, --greater-is-better BOOLEAN &nbsp; Use it to set your score metric options (must be passed together with score-metric option).
* -cpu, --cpu-cores INTEGER &nbsp; Use it to set number of cores in compute target machine.
* -mgb, --memory-gb INTEGER &nbsp; Use it to set memory size in compute target machine.
* -h, --help &nbsp; Show this message and exit.
​
### 2.3. publish-pipeline 
The ```publish-pipeline``` command is used publish your pipeline in azureml. 
​
##### 2.3.1. Arguments:
* PIPELINE_NAME &nbsp; enter your pipeline name to publish
* COMPUTE_TRAGET &nbsp; enter compute target name (if not exists it will create a new one).
​
##### 2.3.2. Options:
* -vs, --vm-size TEXT &nbsp; Use it to set vm size.
* -mnn, --min-nodes INTEGER &nbsp; Use it set min nodes number.
* -mxn, --max-nodes INTEGER &nbsp; Use it set max nodes number.
* -h, --help &nbsp; Show this message and exit.
​
### 2.4. publish-multisteps-pipeline 
The ```publish-multisteps-pipeline``` command is used create a pipeline with multiple steps for running in a flow. 
​
##### 2.4.1. Arguments:
* PIPELINE_NAME &nbsp; enter your pipeline name to publish
​
##### 2.4.2. Options:
* -h, --help &nbsp; Show this message and exit.
