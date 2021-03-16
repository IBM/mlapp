import os

run_file = \
'''from mlapp import MLApp
from config import settings

mlapp = MLApp(settings)


configs = [
    {
        'config_path': "assets/asset_name/configs/asset_name_train_config.json",  # required
        'asset_name': "asset_name",  # optional
        'config_name': "asset_name_config"  # optional (use with .py files)
    }
]


if __name__ == '__main__':
    for config in configs:
        mlapp.run_flow(config.get('asset_name'), config['config_path'], config.get('config_name'))
'''

app_file = \
'''from mlapp import MLApp
from config import settings

mlapp = MLApp(settings)


if __name__ == '__main__':
    mlapp.run_listener()
'''


utilities_file = \
'''#############################################################
#                                                           #
#  Here you can implement common utilities for your assets  #
#                                                           #
#############################################################
'''


empty_config_file = \
'''settings = {

}
'''


default_config_file = \
'''settings = {
    'env_file_path': "''' + os.path.join('env', '<FILENAME>.env') + '''"
}
'''


docker_compose_file = \
'''version: '3.7'

services:
  mq:
    image: "rabbitmq:3-management"
    environment:
      RABBITMQ_DEFAULT_USER: "guest"
      RABBITMQ_DEFAULT_PASS: "guest"
      RABBITMQ_DEFAULT_VHOST: "/"
    hostname: "rabbitmq"
    ports:
      - "15673:15672"
      - "5673:5672"
    labels:
      NAME: "rabbitmq"

  db:
    image: postgres:12.1
    restart: always
    environment:
      POSTGRES_USER: 'postgres'
      POSTGRES_PASSWORD: 'mlapp'
      POSTGRES_DB: 'mlapp'
    ports:
      - '5433:5432'
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  fs:
    image: minio/minio:RELEASE.2019-08-21T19-40-07Z
    volumes:
      - minio-data:/data
    ports:
      - "9001:9000"
    environment:
      MINIO_ACCESS_KEY: minio
      MINIO_SECRET_KEY: minio123
    command: server /data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://minio:9000/minio/health/live"]
      interval: 1m30s
      timeout: 20s
      retries: 3
      start_period: 3m
volumes:
  minio-data:
  postgres-data:
'''

init_sql_file = \
'''CREATE TABLE public.analysis_results (
	model_id uuid NOT NULL,
	asset_name varchar(255) NOT NULL,
	asset_label varchar(255) NULL,
	pipeline varchar(255) NOT NULL,
	properties json NOT NULL,
	metadata json NOT NULL,
	environment varchar(255) NULL,
	created_at timestamptz NULL,
	CONSTRAINT analysis_results_pkey PRIMARY KEY (model_id)
);

CREATE TABLE public.asset_accuracy_monitoring (
	model_id uuid NOT NULL,
	asset_name varchar(255) NOT NULL,
	asset_label_name varchar(255) NOT NULL,
	created_at timestamptz NULL,
	updated_at timestamptz NULL,
	"timestamp" timestamptz NULL DEFAULT CURRENT_TIMESTAMP,
	model_accuracy json NOT NULL
);

CREATE TABLE public.flows (
	flow_id uuid NOT NULL,
	metadata json NOT NULL,
	properties json NOT NULL,
	created_at timestamptz NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.target (
	"timestamp" timestamptz NOT NULL,
	model_id uuid NOT NULL,
	forecast_id uuid NOT NULL,
	"index" varchar(255) NOT NULL,
	y_true float4 NULL,
	y_hat float4 NULL,
	"type" int4 NOT NULL,
	CONSTRAINT target_pkey PRIMARY KEY (model_id, forecast_id, index, type)
);

CREATE TABLE public.jobs (
	id uuid NOT NULL,
	"user" varchar(255) NOT NULL,
	"data" json NULL,
	status_code int4 NULL,
	status_msg varchar(1028) NULL,
	created_at timestamptz NULL,
	updated_at varchar(255) NULL,
	CONSTRAINT tasks_pkey PRIMARY KEY (id)
);
'''

vue_env_config_file = \
'''var env_config = (() => {
  return {
    "VUE_APP_BASE_URL": "http://localhost:3001",
    "VUE_APP_FILE_STORAGE_URL": "http://127.0.0.1:9001",
    "VUE_APP_LOGIN_REQUIRED": "false",
    "VUE_APP_LOGIN_TYPE": "basic",
    "VUE_APP_LOGIN_BACKGROUND": "",
    "VUE_APP_LOGO": ""
  };
})();
'''

env_file = \
'''# RabbitMQ
RABBITMQ_MLAPP_SERVICE_TYPE=rabbitmq
RABBITMQ_MAIN_MQ=true
RABBITMQ_RABBITMQ_HOSTNAME=127.0.0.1
RABBITMQ_RABBITMQ_PORT=5673
RABBITMQ_RABBITMQ_CONNECTION_TIMEOUT=15

# Postgres
POSTGRES_MLAPP_SERVICE_TYPE=postgres
POSTGRES_MAIN_DB=true
POSTGRES_POSTGRES_DATABASE_NAME=mlapp
POSTGRES_POSTGRES_HOSTNAME=127.0.0.1
POSTGRES_POSTGRES_PORT=5433
POSTGRES_POSTGRES_USER_ID=postgres
POSTGRES_POSTGRES_PASSWORD=mlapp

# Minio
MINIO_MLAPP_SERVICE_TYPE=minio
MINIO_MAIN_FS=true
MINIO_MINIO_ENDPOINT=127.0.0.1
MINIO_MINIO_ACCESS_KEY=minio
MINIO_MINIO_SECRET_KEY=minio123
MINIO_MINIO_PORT=9001
'''


azureml_env_file = \
'''# AML Storage handler
AML-STORAGE_MLAPP_SERVICE_TYPE=azureml_model_storage
AML-STORAGE_MAIN_FS=true

# azureml-queue azureml-queue service
AZUREML-QUEUE_MLAPP_SERVICE_TYPE=azureml_queue
AZUREML-QUEUE_MAIN_MQ=true
'''


# This file expects for asset_name argument.
model_manager_file = \
'''from mlapp.managers import ModelManager, pipeline


class {}ModelManager(ModelManager):
    @pipeline
    def train_model(self, data):
        raise NotImplementedError()

    @pipeline
    def forecast(self, data):
        raise NotImplementedError()
    
    @pipeline
    def refit(self, data):
        raise NotImplementedError()
'''

# This file expects for asset_name argument.
data_manager_file = \
'''from mlapp.managers import DataManager, pipeline


class {}DataManager(DataManager):
    # -------------------------------------- train pipeline -------------------------------------------
    @pipeline
    def load_train_data(self, *args):
        raise NotImplementedError("should return data")
    
    @pipeline
    def clean_train_data(self, data):
        return data
    
    @pipeline
    def transform_train_data(self, data):
        return data

    # ------------------------------------ forecast pipeline -------------------------------------------
    @pipeline
    def load_forecast_data(self, *args):
        raise NotImplementedError("should return data")
    
    @pipeline
    def clean_forecast_data(self, data):
        return data

    @pipeline
    def transform_forecast_data(self, data):
        return data
    
    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()

'''

# This file expects to replace `<ASSET_NAME>` in asset_name.
train_config_file = \
'''{
  "pipelines_configs": [
    {
      "data_settings": {

      },
      "model_settings": {

      },
      "job_settings": {
        "asset_name": "<ASSET_NAME>",
        "pipeline": "train"
      }
    }
  ]
}
'''

# This file expects to replace `<ASSET_NAME>` in asset_name.
train_config_file_with_flow = \
'''{
  "pipelines_configs": [
    {
      "data_settings": {
        
      },
      "model_settings": {
        
      },
      "job_settings": {
        "asset_name": "<ASSET_NAME>",
        "pipeline": "train"
      }
    }
  ],
  "flow_config": {
    
  },
  "job_settings": {
    "task_type": "flow"
  }
}
'''

# This file expects to replace `<ASSET_NAME>` in asset_name.
forecast_config_file = \
'''{
  "pipelines_configs": [
    {
      "data_settings": {
        
      },
      "model_settings": {
        
      },
      "job_settings": {
        "asset_name": "<ASSET_NAME>",
        "pipeline": "forecast",
        "model_id": "latest"
      }
    }
  ]
}
'''

# This file expects to replace `<ASSET_NAME>` in asset_name.
forecast_config_file_with_flow = \
'''{
  "pipelines_configs": [
    {
      "data_settings": {

      },
      "model_settings": {
      },
      "job_settings": {
        "asset_name": "<ASSET_NAME>",
        "pipeline": "forecast",
        "model_id": "latest"
      }
    }
  ],
  "flow_config": {},
  "job_settings": {
    "task_type": "flow"
  }
}
'''


gitignore_file = \
'''*.idea/
*.pyc
.DS_Store
my-vern
my_venv
*.log
*.pkl
*.csv
output/
temporary_output/
test_output/
metastore_db/
cache/
venv/
dist/
build/
*egg-info
*.env*
.env*
'''

dockerignore_file = \
'''.dockerignore
Dockerfile
db.sqlite3
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
coverage.xml
*,cover
*.log
.git
*.idea/
.DS_Store
my-vern
my_venv
venv/
dist/
output/
temporary_output/
test_output/
build/
*egg-info
telepresence.log
'''

amlignore_file = '''*.idea/
*.pyc
.DS_Store
my-vern
my_venv
output/
temporary_output/
test_output/
metastore_db/
cache/
venv/
dist/
build/
*egg-info

'''