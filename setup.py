import imp
import os
from setuptools import setup, find_packages
from os import path

# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

version = imp.load_source('mlapp.version', os.path.join('mlapp', 'version.py')).VERSION
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = [x.strip() for x in f if x.strip()]

setup(
    name='mlapp',  # Required
    version=version,  # Required
    description='IBM Services Framework for ML Applications Python 3 framework for building robust, '
                'production-ready machine learning applications. '
                'Official ML accelerator within the larger RAD-ML methodology.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ibm/mlapp',
    author='IBM',
    author_email='tomer.galula@ibm.com, tal.waitzenberg@ibm.com, michael.chein@ibm.com, erez.nardia@ibm.com, '
                 'annaelle.cohen@ibm.com, katzn@us.ibm.com',
    keywords=['mlapp', 'ibm', 'machine-learning', 'auto-ml'],
    packages=find_packages(exclude=['tests', 'docs', 'venv']),  # Required
    license="Apache License 2.0",
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License'
    ],
    python_requires='>=3.6',

    install_requires=requires,

    extras_require={  # Optional
        'rabbitmq': ['pika'],
        'minio': ['minio'],
        'mysql': ['PyMySQL'],
        'snowflake': ['snowflake-sqlalchemy'],
        'azure-servicebus': ['azure-servicebus<=0.50.3'],
        'kafka': ['kafka-python'],
        'kafka-kerberos': ['confluent_kafka'],
        'boto3': ['boto3'],
        'ibm-boto3': ['ibm-cos-sdk'],
        'azure-storage-blob': ['azure-storage-blob<=2.1.0'],
        'postgres': ['pg8000<=1.16.5'],
        'livy': ['livy'],
        'mssql': ['pyodbc'],
        'pyspark': ['pyspark'],
        'aml': [
            'azureml-sdk'
        ],
        'mlcp': [
            'pika',
            'pg8000<=1.16.5',
            'minio'
        ]
    },
    # package_data={
    #     'mlapp': ['mlapp/cli/script.sh'],
    # },
    # data_files=[('my_data', ['data/data_file'])],  # Optional
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'mlapp=mlapp.cli:cli',
        ],
    },

    project_urls={
        'Bug Reports': 'https://github.com/ibm/mlapp/issues',
        'Wiki Page': 'https://mlapp-docs.s3-web.us-south.cloud-object-storage.appdomain.cloud',
        'Crash Course': 'https://mlapp-docs.s3-web.us-south.cloud-object-storage.appdomain.cloud'
                        '/crash-course/introduction',
        'Source': 'https://github.com/ibm/mlapp'
    },
)
