# Connecting with Databricks

If you have a Python 3.5 environment installed in your OS you can skip to [this step](#install-required-packages).

## Instructions for Mac OS X

### Install _pyenv_
_pyenv_ is a python versioning software and is used in this case for installing python 3.5, as python 3.5 is required by Databricks.
```bash
brew install pyenv
brew install pyenv-virtualenv
```

### Update your shell with the new installations
```bash
exec $SHELL
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

### Tkinter Python binding to the Tk GUI toolkit
You must have **tcl-tk** installed. you can check it with the command: 
```bash
brew search tcl-tk
```
In case you don't, use this command to install: 
```bash
brew install tcl-tk
````

Run these commands to export Tkinter for your python 3.5 installation:
```bash
export LDFLAGS="-L/usr/local/opt/tcl-tk/lib"
export CPPFLAGS="-I/usr/local/opt/tcl-tk/include"
export PATH=$PATH:/usr/local/opt/tcl-tk/bin
```

### Install Python 3.5.2
```bash
pyenv install 3.5.2
```

###  Add a new virtual environment and activate it
```bash
pyenv virtualenv 3.5.2 dbconnect
pyenv activate dbconnect
```

### Configure PyCharm's Python interpreter to use this new environment 
1. Open PyCharm's `Preferences...`.
2. Select the `Project Interpreter` tab.
3. Click the gear icon and select `Add...` to add a new project interpreter.
4. Select the `Virtualenv Environment` tab. 
5. Choose `Existing environment` and see that the Interpreter path is pointing to `~/.pyenv/versions/dbconnect/bin/python`.

## Install required packages
```bash
pip install git+ssh://git@github.com/ibm/mlapp.git
pip install -U databricks-connect==5.5.*
```

## Go through the steps in the official Databricks Connect guide
[https://docs.azuredatabricks.net/dev-tools/db-connect.html](https://docs.azuredatabricks.net/dev-tools/db-connect.html)

> **Note**: Some of the steps in the _Databricks Connect_ guide were done in this wiki. Skip through the following steps:
> - Requirement of installing a Python 3.5 environment.
> - Step 1: Install the client.


## Setup _MLApp_ in your new project folder
```bash
mlapp init
mlapp environment init dev
mlapp services add databricks
```

> **Note**: while running these commands you should have the _dbconnect_ virtual environment activated.

In case an error with running command `mlapp` occurs, run these:
```bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```
> **Note**: swap language to another as you see fit, `en_US` -> `es_ES`

## Using Databricks in _MLApp_
When you previously added the _databricks_ service via the _MLApp CLI_, you picked a name as well. 

The followings are available options for using the _databricks_ service in your **Data Manager**:

#### Import Spark Handler
Import the Spark handler:
```python
from mlapp.handlers.instance import spark_handler
```

#### Running Spark SQL Queries:
```python
spark_handler('<databricks_service_name>').exec_query(query, params=None)
```
#### Loading CSV Files:
```python
spark_handler('<databricks_service_name>').load_csv_file(file_path, sep=',', header=True, toPandas=False, **kwargs)
```


