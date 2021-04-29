# Control Panel - Local Setup

The full application example is done on your local computer. It includes connecting with a database, file storage and message queue via Docker.

## Prerequisites
- Initiated MLApp project: `mlapp init`.
- Asset available in the MLApp project (e.g. `mlapp boilerplates install crash_course`) 
- Docker, Docker Compose (version >=1.24)
- Python library for [RabbitMQ](https://github.com/pika/pika): `pip install pika`
- Python library for [PostgreSQL(version<=1.16.5)](https://github.com/tlocke/pg8000): `pip install pg8000<=1.16.5`
- Python library for [Minio](https://github.com/minio/minio-py): `pip install minio`

!!! tip "Python's required libraries for the Control Panel can be installed via pip's extras option"

    Use: `pip install "mlapp[cp]"`

## Service Types

### Database

The database will hold any metadata of the models which is similar to the `metadata.json` output when running models locally in MLApp.

The database supports a job table to monitor runs, and a table for model drift monitoring. 

Predictions can be stored in the database as well.

### File Storage

The file storage will hold the model objects, logs and configs.

### Message Queue

The message queue will be a connecting point of the python worker running your code. 

Sending config messages to the queue will be picked up by the python worker.


## Setup

You can install the example via cli command:
```shell
mlapp cp setup
```

!!! tip "Initiated Files at Your Root Project"

    **deployment/docker-compose.yaml** - configuration of the docker containers.

    **deployment/env-config.js** - configuration file for the control panel UI.

## Run `docker-compose`

Run the following commands from your root project:
```shell
cd deployment
docker-compose up
```

!!! tip "Docker Containers"
    
    This command will run the containers configured in the **docker-compose.yaml** file. 

    A **PostgreSQL** database, a **Minio** file storage, a **RabbitMQ** message queue and a **Redis** session will all be running in on your Docker environment.

!!! note "Stopping Containers"

    Be sure to run command `docker-compose down` when you're finished with this example to stop the containers from running on Docker.

    You can use `-v` flag to delete the created database and file storage persistent volumes in your local docker hub.

!!! note "MLApp CLI"
    
    There are equivalent commands in the MLApp CLI for running the above: `mlapp cp start`, `mlapp cp stop`. 

## Test Your Application

### Run `app.py`

Run the `app.py` file at your root project.

### Open Control Panel

Open in your browser the following url [http://localhost:8081](http://localhost:8081).

For more information on using the control panel check [here](/integrations/control-panel/introduction).
