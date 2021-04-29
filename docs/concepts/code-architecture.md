# Code Architecture

We will detail the important files and folders in the architecture.

When setting up a new project with _MLApp_ library installed you can initiate the code architecture with the `mlapp init` command:

```text
Usage: mlapp init [OPTIONS]

  Creates an initial project structure

Options:
  -cp, --control-panel            Flag that includes the MLApp control panel in
                                  your project.
  -g, --gitignore                 Flag that adds .gitignore file into your
                                  project.
  -d, --dockerignore              Flag that adds .dockerignore file into your
                                  project.
  -f, --force                     Flag force init if project folder is not
                                  empty.
  -h, --help                      Show this message and exit.
```

## 1. Key Files

### 1.1. run.py

This file is the key file to run assets in your local environment. 

All assets are run by configurations and you can set up which configuration to run by editing the `configs` list variable:

```python
configs = [{
    "config_path": "assets/crash_course/configs/crash_course_train_flow_config.json",
    "asset_name": "crash_course",
    "config_name": "crash_course_config"
}]
```

> Note: each item in the `configs` variable is a _dict_ type with the following keys:
>
> **config_path**: path to the configuration file, can be a **.json** or **.py** file.
>
> **asset_name**: this is the name of the asset (required only if configuration is a python file).
>
> **config_name**: this is the name of the variable _dict_ containing the config in the **.py** file (required only if configuration is a python file).

### 1.2. requirements.txt

Adds consistency to the environment between Data Scientists who are developing and sharing code in the project.

Put any libraries you are using in your project and their version. 

For more general information on `requirement.txt` file you can check [here](https://pip.readthedocs.io/en/1.1/requirements.html).


### 1.3. .gitigonre

Put it any files you don’t want to in your VCS (version control system).

Add the `-g` or `--gitignore` flag to the `mlapp init` command to create this file.

### 1.4. .dockerignore

Put it any files you don’t want in the Docker images that will be built for production.

Add the `-d` or `--dockerignore` flag to the `mlapp init` command to create this file. 

### 1.5. app.py

Key file when running project in "Dockerized/Containerized" form or when using the Control Panel.

### 1.6. config.py

This file is automatically modified and holds the configuration of MLApp.

It holds configurations such as `env_file_path` which specifies the path to your environment file.

## 2. Key Directories

### 2.1. assets

This directory holds your assets' code. Each asset is separated into a different directory by it's asset name.

Files and directories that are in an asset: 

- **<_asset_name_>_data_manager.py**
- **<_asset_name_>_model_manager.py**
- **configs/<_asset_name_>_train_config.json** 
- **configs/<_asset_name_>_forecast_config.json**.

### 2.2. common

All logic you implement and want to share between different assets in your project.


### 2.3. env

Holds environment files with credentials to external services. Handled via MLApp CLI.

### 2.4. deployment

Folder contains files for running MLApp on Control Panel/AzureML or any other platform.

### 2.5. data

Contains data files for running local tests and debugging.


### 2.6. output

Contains all output files from your local runs – log, pickle, csv, json, images, etc.