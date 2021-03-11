

# MLApp &middot; [![pip version](https://img.shields.io/pypi/v/mlapp?color=success)](https://pypi.python.org/pypi/mlapp/) [![Build Status](https://travis-ci.com/IBM/mlapp.svg?branch=master)](https://travis-ci.com/IBM/mlapp) [![License](https://img.shields.io/badge/license-Apache-blue.svg)](https://github.com/IBM/mlapp/blob/master/LICENSE)

MLApp is a Python library for building machine learning and AI solutions that are consistent, integrated and production-ready.

- **Versatile**: Applicable towards a wide variety of use cases including statistical modeling, machine learning, deep learning and even optimization. Developers can install and use their favorite Python packages (scikit-learn, xgboost etc.) easily as part of their MLApp projects.
- **Project scaffolding**: Generates opinionated file structure that enforces modern engineering standards and improves readability across solutions.
- **Embedded with MLOps**: Standardizes the way models and their metadatas are registered, stored and deployed.
- **Asset boilerplates**: Pre-built model templates that can be easily customized to accelerate development of common use cases.
- **Data science utilities**: Extendable set of utilities (feature selection, autoML and other areas) increasing developer productivity.
- **Connectors**: Easily connect to common data and analytics services.
- **Deployment integration**: Applications built using MLApp can easily be deployed on platforms such as Kubernetes, Azure Machine Learning and others.

## Getting started

Install MLApp via pip:

```
pip install mlapp
```

Navigate to an empty project folder and generate the project scaffold:

```
mlapp init
```

Install a working example using boilerplates:

```
mlapp boilerplates install basic_regression
```

Update the run.py file in your project directory to point to the Basic Regression asset that you just installed:

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

Congrats! You've trained your first model in MLApp. Take a look at the output directory to see the results.

## Next steps
A great place to start is the [crash course](https://mlapp-docs.s3-web.us-south.cloud-object-storage.appdomain.cloud/crash-course/introduction).

You should also check out the full [project documentation](https://mlapp-docs.s3-web.us-south.cloud-object-storage.appdomain.cloud).

## Contributing
We welcome contributions from the community. Please refer to [CONTRIBUTING](./CONTRIBUTING.md) for more information.