# MLApp Quickstart

## Installation

MLApp works with Python 3.6.4 or higher - you may want to start by checking your Python installation version:

```
python --version
```

If that looks good, go ahead and install MLApp via pip:

```
pip install mlapp
```

## Create a project

Now that MLApp is installed, lets create a project. Navigate to an empty project folder and generate the project file structure:

```
mlapp init
```

And then create an empty modeling asset:

```
mlapp assets create new_asset
```

This generates an empty modeling asset within the ```assets``` directory - to instead install a working example using boilerplates:

```
mlapp boilerplates install basic_regression
```

## Train a model

Now lets train a model! To do so, first update the ```run.py``` file in your project directory to point to the basic regression asset that you just installed:

```
configs = [
    {
        'config_path': "assets/basic_regression/configs/basic_regression_train_config.py",
        'asset_name': "basic_regression",
        'config_name': "basic_regression_config"
    }
]
```

Execute the run.py script:

```
python run.py
```

Congrats! You've trained your first model in MLApp. You should see the model execution logs in your terminal. When the process is complete, take a look at the output directory to see the results.

## Installing MLApp extensions

MLApp Extensions are additional sets of functionality that need to be installed separately. to do so, use the `pip install`. For example, to install the extension that enables deployment in Azure Machine Learning:

```
pip install "mlapp[aml]"
```

> Refer to the _extras_require_ attribute in your `setup.py` file for more information on 3rd party libraries we use to connect with different services.

## Where to next
A great place to look next is the [MLApp Crash Course](/crash-course/introduction).
