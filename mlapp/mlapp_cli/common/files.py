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
    'env_file_path': 'env/<FILENAME>.env'
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

  redis:
    image: bitnami/redis:latest
    volumes:
      - redis-data:/bitnami/redis/data
    environment:
      - ALLOW_EMPTY_PASSWORD=yes

  redis-sentinel:
    image: 'bitnami/redis-sentinel:latest'
    depends_on:
      - redis
    environment:
      - REDIS_MASTER_HOST=redis
      - REDIS_MASTER_PORT_NUMBER=6379
      - REDIS_MASTER_SET=mymaster
      - REDIS_SENTINEL_PORT_NUMBER=26379
    ports:
      - '26379:26379'

  frontend:
    image: radml/vuejs:2.0.2
    volumes:
      - ./env-config.js:/usr/share/nginx/html/env-config.js
    ports:
      - "8081:80"

  backend:
    image: radml/nodejs:2.0.2
    environment:
      - CORS=http://localhost:8081
      - DB_TYPE=knex
      - DB_ADAPTER=postgres
      - DB_HOST=db
      - DB_USER=postgres
      - DB_PASSWORD=mlapp
      - DB_PORT=5432
      - DB_NAME=mlapp
      - FS_TYPE=minio
      - FS_ENDPOINT=fs
      - FS_ACCESSKEY=minio
      - FS_SECRETKEY=minio123
      - FS_PORT=9000
      - MQ_TYPE=rabbitmq
      - MQ_ENDPOINT=amqp://mq:5672
      - APP_LOGIN_REQUIRED=false
      - APP_IS_HTTPS=false
      - APP_LOGIN_TYPE=basic
      - SESSION_TYPE=redis
      - REDIS_HOST=127.0.0.1
      - REDIS_PORT=26379
      - SENTINEL_ENDPOINT=master
    ports:
      - "3001:3000"
    depends_on:
      - "db"
      - "mq"
      - "fs"
      - "redis-sentinel"

volumes:
  minio-data:
  postgres-data:
  redis-data:
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
'''from mlapp.managers import ModelManager
from mlapp.utils import pipeline


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
'''from mlapp.managers import DataManager
from mlapp.utils import pipeline


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