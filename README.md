

# MLApp &middot; [![pip version](https://img.shields.io/pypi/v/mlapp?color=success)](https://pypi.python.org/pypi/mlapp/) [![Build Status](https://travis-ci.com/IBM/mlapp.svg?branch=master)](https://travis-ci.com/IBM/mlapp) [![License](https://img.shields.io/badge/license-Apache-blue.svg)](https://github.com/IBM/mlapp/blob/master/LICENSE)

MLApp is a Python library for building scalable data science solutions that meet modern software engineering standards.

MLApp was built and hardened in an enterprise context, to solve scalability issues for mid-size to Fortune 50 companies. It is applicable to a variety of data science use cases including machine learning, deep learning, NLP and optimization.

- **Embedded MLOps**: Standardizes the way models and their metadatas are registered, stored and deployed.
- **Project scaffolding**: Generates an opinionated project file structure that enforces modern engineering standards and improves readability and documentation across solutions.
- **Boilerplates**: Includes a library of pre-built model templates that can be easily customized to accelerate development of common use cases.
- **Utilities**: Includes an extendable set of utilities that increase developer productivity - including functions for selecting features and optimizing hyperparameters.
- **Connectors**: Allows developers to easily integrate their projects with common data and analytics services.
- **Deployment integration**: Applications built using MLApp can easily be deployed on common open and proprietary platforms, including Kubernetes and Azure Machine Learning.

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