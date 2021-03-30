# Full Application Example

The full application example is done on your local computer. It includes connecting with a database, file storage and message queue via Docker.

## Prerequisites
- Initiated MLApp project: `mlapp init`.
- Asset available in the MLApp project: `mlapp boilerplates install crash_course` 
- Docker
- Docker Compose (version >=1.24)
- Python library for [RabbitMQ](https://github.com/pika/pika): `pip install pika`
- Python library for [PostgreSQL(version<=1.16.5)](https://github.com/tlocke/pg8000): `pip install pg8000<=1.16.5`
- Python library for [Minio](https://github.com/minio/minio-py): `pip install minio`

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
mlapp mlcp setup
```

!!! tip "Initiated Files at Your Root Project"

    **deployment/docker-compose.yaml** - configuration of the docker containers.

    **deployment/init.sql** - tables schemas for the database.

## Run `docker-compose`

Run the following commands from your root project:
```shell
cd deployment
docker-compose up
```

!!! tip "Docker Containers"
    
    This command will run the containers configured in the **docker-compose.yaml** file. 

    A **PostgreSQL** database, a **Minio** file storage and a **RabbitMQ** message queue will all be running in on your Docker environment.

!!! note "Stopping Containers"

    Be sure to run command `docker-compose down` when you're finished with this example to stop the containers from running on Docker.

!!! note "MLApp CLI"
    
    There are equivalent commands in the MLApp CLI for running the above: `mlapp mlcp start`, `mlapp mlcp stop`. 

## Test Your Application

### Run `app.py`

Run the `app.py` file at your root project.

### Send Config from Message Queue

1. Open in your browser the following url [http://localhost:15673](http://localhost:15673).

2. Login with the default username:`guest` and password:`guest` for the RabbitMQ docker container image.

3. Click on the **Queues** tab at the top and select the available queue `analysis_general_listen`.

4. Open up the `Publish message` accordion and insert a configuration in the **Payload** text area box.

5. Click button **Publish message**.

## Explore Results

Now your config should run and results stored in all the external services when finished.

For the **Minio** service results you can view them via your browser at [http://localhost:9001](http://localhost:9001) with access key:`minio` and secret key: `minio123`.

For the database results you'll need a software that connects to **PostgreSQL** to view the results.


