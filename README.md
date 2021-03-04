

# MLApp &middot; [![pip version](https://img.shields.io/pypi/v/mlapp?color=success)](https://pypi.python.org/pypi/mlapp/) [![Build Status](https://travis-ci.com/IBM/mlapp.svg?branch=master)](https://travis-ci.com/IBM/mlapp) [![License](https://img.shields.io/badge/license-Apache-blue.svg)](https://github.com/IBM/mlapp/blob/master/LICENSE)

MLApp is a Python library for building machine learning and AI solutions that are consistent, integrated and production-ready.

- **Project scaffolding**: Generates opinionated file structure that enforces modern engineering standards and improves readability across solutions
- **Embedded with MLOps**: Standardize the way models and their metadatas are registered, stored and deployed
- **Asset boilerplates**: Pre-built model templates that can be easily customized to accelerate development of common use cases
- **Data science utilities**: Extendable set of utilities (feature selection, autoML and other areas) increasing developer productivity
- **Connectors**: Easily connect to common data and analytics services
- **Deployment integration**: Applications built using MLApp can easily be deployed on platforms such as Kubernetes, Azure Machine Learning and others

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
mlapp boilerplates install crash_course
```

Update the run.py file in your project directory to point to the Crash Course asset and configuration file:

```
configs = [
    {
        'config_path': "assets/crash_course/configs/crash_course_train_config.json",
        'asset_name': "crash_course",
        'config_name': "crash_course_config"
    }
]
```

Execute the run.py file to train your first model:

```
python3 run.py
```

Congrats! You've trained your first model in MLApp. Take a look at the output directory to see the results.

## Next steps
A great place to start is the [crash course](https://mlapp-docs.s3-web.us-south.cloud-object-storage.appdomain.cloud/crash-course/introduction) which goes into more detail about the example you completed above.

You should also check out the full [project documentation](https://mlapp-docs.s3-web.us-south.cloud-object-storage.appdomain.cloud).

## Contributing
We welcome contributions from the community. Please refer to [CONTRIBUTING](./CONTRIBUTING.md) for more information.