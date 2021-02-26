from mlapp.handlers.databases.mssql_handler import MssqlHandler
from mlapp.handlers.databases.mysql_handler import MySQLHandler
from mlapp.handlers.databases.postgres_handler import PostgresHandler
from mlapp.handlers.databases.snowflake_handler import SnowflakeHandler

try:
    from mlapp.handlers.file_storages.minio_storage_handler import MinioStorageHandler
except Exception as e:
    MinioStorageHandler = None

try:
    from mlapp.handlers.message_queues.rabbitmq_handler import RabbitMQHandler
except Exception as e:
    RabbitMQHandler = None

try:
    from mlapp.handlers.file_storages.azure_blob_handler import AzureBlobHandler
except Exception as e:
    AzureBlobHandler = None

try:
    from mlapp.handlers.message_queues.azure_service_bus_handler import AzureServicesBusHandler
except Exception as e:
    AzureServicesBusHandler = None

try:
    from mlapp.handlers.spark.spark_handler import SparkHandler
except Exception as e:
    SparkHandler = None

try:
    from handlers.hive_handler import HiveHandler
except Exception as e:
    HiveHandler = None

try:
    from mlapp.handlers.spark.livy_handler import LivyHandler
except Exception as e:
    LivyHandler = None

try:
    from mlapp.handlers.file_storages.boto_storage_handler import BotoStorageHandler
except Exception as e:
    BotoStorageHandler = None

try:
    from mlapp.handlers.file_storages.ibm_boto3_storage_handler import IBMBoto3StorageHandler
except Exception as e:
    IBMBoto3StorageHandler = None

try:
    from mlapp.handlers.message_queues.kafka_handler import KafkaHandler
except Exception as e:
    KafkaHandler = None

try:
    from mlapp.handlers.message_queues.kafka_kerberos_handler import KafkaKerberosHandler
except Exception as e:
    KafkaKerberosHandler = None

try:
    from mlapp.handlers.file_storages.aml_model_storage_handler import AmlModelStorageHandler
except Exception as e:
    AmlModelStorageHandler = None

try:
    from mlapp.handlers.file_storages.aml_run_storage_handler import AmlRunStorageHandler
except Exception as e:
    AmlRunStorageHandler = None

try:
    from mlapp.handlers.message_queues.aml_queue import AMLQueue
except Exception as e:
    AMLQueue = None

from mlapp.env_loader import EMPTY_STRING, ZERO


def environment_services(env=None):
    """
    Holds the mapping between the environment file and the input required for each handler
    :param env: Env class - from environs library
    """
    return {
        # default mysql db settings for Kubernetes deployment. See ML App Wiki documentation
        "mysql": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_DB', default=False),
            "handler": MySQLHandler,
            "type": "database",
            "settings": {
                "database_name": env.str(x.upper() + '_MYSQL_DATABASE_NAME', default=EMPTY_STRING),
                "hostname": env.str(x.upper() + '_MYSQL_HOSTNAME', default=EMPTY_STRING),
                "password": env.str(x.upper() + '_MYSQL_PASSWORD', default=EMPTY_STRING),
                "port": env.int(x.upper() + '_MYSQL_PORT', default=ZERO),
                "user_id": env.str(x.upper() + '_MYSQL_USER_ID', default=EMPTY_STRING),
                "options": {}
            }
        },
        # default minio file storage settings for Kubernetes deployment. See ML App Wiki documentation
        "minio": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_FS', default=False),
            "handler": MinioStorageHandler,
            "type": "file_storage",
            "settings": {
                'endPoint': env.str(x.upper() + '_MINIO_ENDPOINT', default=EMPTY_STRING),
                'accessKey': env.str(x.upper() + '_MINIO_ACCESS_KEY', default=EMPTY_STRING),
                'secretKey': env.str(x.upper() + '_MINIO_SECRET_KEY', default=EMPTY_STRING),
                'port': env.str(x.upper() + '_MINIO_PORT', default=EMPTY_STRING),
                'secure': env.bool(x.upper() + '_MINIO_SECURE', default=False),
                'region': env.str(x.upper() + '_MINIO_REGION', default=EMPTY_STRING)
            }
        },
        "azure_blob": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_FS', default=False),
            "type": "file_storage",
            "handler": AzureBlobHandler,
            "settings": {
                'accountName': env.str(x.upper() + '_AZURE_BLOB_ACCOUNT_NAME', default=EMPTY_STRING),
                'accountKey': env.str(x.upper() + '_AZURE_BLOB_ACCOUNT_KEY', default=EMPTY_STRING),
            }
        },
        "azureml_model_storage": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_FS', default=False),
            "type": "file_storage",
            "handler": AmlModelStorageHandler,
            "settings": {
                "local_storage_path": settings.get('local_storage_path'),
                "temporary_storage_path": settings.get('temporary_storage_path'),
                "file_store_buckets": settings.get('file_store_buckets')
            }
        },
        "azureml_run_storage": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_FS', default=False),
            "type": "file_storage",
            "handler": AmlRunStorageHandler,
            "settings": {
                "local_storage_path": settings.get('local_storage_path'),
                "temporary_storage_path": settings.get('temporary_storage_path'),
                "file_store_buckets": settings.get('file_store_buckets'),
                "database_name": env.str(x.upper() + '_AML_RUN_STORAGE_DATABASE_NAME', default=EMPTY_STRING),
                "hostname": env.str(x.upper() + '_AML_RUN_STORAGE_HOSTNAME', default=EMPTY_STRING),
                "password": env.str(x.upper() + '_AML_RUN_STORAGE_PASSWORD', default=EMPTY_STRING),
                "port": env.int(x.upper() + '_AML_RUN_STORAGE_PORT', default=ZERO),
                "user_id": env.str(x.upper() + '_AML_RUN_STORAGE_USER_ID', default=EMPTY_STRING),
                "options": {}
            }
        },
        "azure_service_bus": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_MQ', default=False),
            "type": "message_queue",
            "handler": AzureServicesBusHandler,
            "settings": {
                'hostname': env.str(x.upper() + '_AZURE_SERVICE_BUS_HOSTNAME', default=EMPTY_STRING),
                'shared_access_key_name':
                    env.str(x.upper() + '_AZURE_SERVICE_BUS_SHARED_ACCESS_KEY_NAME', default=EMPTY_STRING),
                'shared_access_key':
                    env.str(x.upper() + '_AZURE_SERVICE_BUS_SHARED_ACCESS_KEY', default=EMPTY_STRING),
                'entity_path': env.str(x.upper() + '_AZURE_SERVICE_BUS_ENTITY_PATH', default=EMPTY_STRING),
            }
        },
        "mssql": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_DB', default=False),
            "type": "database",
            "handler": MssqlHandler,
            "settings": {
                "database_name": env.str(x.upper() + '_MSSQL_DATABASE_NAME', default=EMPTY_STRING),
                "hostname": env.str(x.upper() + '_MSSQL_HOSTNAME', default=EMPTY_STRING),
                "password": env.str(x.upper() + '_MSSQL_PASSWORD', default=EMPTY_STRING),
                "port": env.int(x.upper() + '_MSSQL_PORT', default=ZERO),
                "user_id": env.str(x.upper() + '_MSSQL_USER_ID', default=EMPTY_STRING),
                "options": {}
            }
        },
        "snowflake": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_DB', default=False),
            "type": "database",
            "handler": SnowflakeHandler,
            "settings": {
                "account": env.str(x.upper() + '_SNOWFLAKE_ACCOUNT', default=EMPTY_STRING),
                "user": env.str(x.upper() + '_SNOWFLAKE_USER', default=EMPTY_STRING),
                "password": env.str(x.upper() + '_SNOWFLAKE_PASSWORD', default=EMPTY_STRING),
                "database": env.str(x.upper() + '_SNOWFLAKE_DATABASE', default=EMPTY_STRING),
                "schema": env.str(x.upper() + '_SNOWFLAKE_SCHEMA', default=EMPTY_STRING),
                "warehouse": env.str(x.upper() + '_SNOWFLAKE_WAREHOUSE', default=EMPTY_STRING),
                "role": env.str(x.upper() + '_SNOWFLAKE_ROLE', default=EMPTY_STRING),
                "options": {}
            }
        },
        "postgres": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_DB', default=False),
            "type": "database",
            "handler": PostgresHandler,
            "settings": {
                "database_name": env.str(x.upper() + '_POSTGRES_DATABASE_NAME', default=EMPTY_STRING),
                "hostname": env.str(x.upper() + '_POSTGRES_HOSTNAME', default=EMPTY_STRING),
                "password": env.str(x.upper() + '_POSTGRES_PASSWORD', default=EMPTY_STRING),
                "port": env.int(x.upper() + '_POSTGRES_PORT', default=ZERO),
                "user_id": env.str(x.upper() + '_POSTGRES_USER_ID', default=EMPTY_STRING),
                "ssl": env.bool(x.upper() + '_POSTGRES_SSL', default=False),
                "options": {}
            }
        },
        "databricks": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_SPARK', default=False),
            "type": "spark",
            "handler": SparkHandler,
            "settings": {
                "spark.databricks.service.address": env.str(x.upper() + '_DATABRICKS_ADDRESS', default=EMPTY_STRING),
                "spark.databricks.service.token": env.str(x.upper() + '_DATABRICKS_TOKEN', default=EMPTY_STRING),
                "spark.databricks.service.clusterId": env.str(x.upper() + '_DATABRICKS_CLUSTER_ID', default=EMPTY_STRING),
                "spark.databricks.service.port": env.int(x.upper() + '_DATABRICKS_PORT', default=ZERO),
                "spark.databricks.service.orgId": env.int(x.upper() + '_DATABRICKS_ORGANIZATION_ID', default=ZERO),
                "enable_hive": env.str(x.upper() + '_ENABLE_HIVE', default=False),
            }
        },
        "spark": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_SPARK', default=False),
            "type": "spark",
            "handler": SparkHandler,
            "settings": {
                # cluster
                "spark.sql.warehouse.dir":  env.str(x.upper() + '_SPARK_SQL_WAREHOUSE_DIR', default=EMPTY_STRING),
                "hive.metastore.warehouse.dir":  env.str(x.upper() + '_SPARK_HIVE_METASTORE_WAREHOUSE_DIR', default=EMPTY_STRING),
                "hive.metastore.uris": env.str(x.upper() + '_SPARK_HIVE_METASTORE_URIS', default=EMPTY_STRING),
                "enable_hive": env.bool(x.upper() + '_SPARK_ENABLE_HIVE', default=False),
                "spark.driver.host":  env.str(x.upper() + '_SPARK_DRIVER_HOST', default=EMPTY_STRING),
                "spark.app.id":  env.str(x.upper() + '_SPARK_APP_ID', default=EMPTY_STRING),
                "spark.sql.catalogImplementation":  env.str(x.upper() + '_SPARK_SQL_CATALOG_IMPLEMENTATION', default=EMPTY_STRING),
                "spark.rdd.compress":  env.bool(x.upper() + '_SPARK_RDD_COMPRESS', default=False),
                "spark.serializer.objectStreamReset": env.int(x.upper() + '_SPARK_SERIALIZER_OBJECT_STREAM_RESET', default=ZERO),
                "spark.master": env.str(x.upper() + '_SPARK_MASTER', default=EMPTY_STRING),
                "spark.executor.id": env.str(x.upper() + '_SPARK_EXECUTOR_ID', default=EMPTY_STRING),
                "spark.driver.port": env.int(x.upper() + '_SPARK_DRIVER_PORT', default=ZERO),
                "spark.submit.deployMode": env.str(x.upper() + '_SPARK_SUBMIT_DEPLOY_MODE', default=EMPTY_STRING),
                "spark.app.name": env.str(x.upper() + '_SPARK_APP_NAME', default=EMPTY_STRING),
                "spark.ui.showConsoleProgress": env.bool(x.upper() + '_SPARK_UI_SHOW_CONSOLE_PROGRESS', default=True),

                # database
                "driver":  env.str(x.upper() + '_SPARK_DRIVER', default=EMPTY_STRING),
                "connector_type":  env.str(x.upper() + '_SPARK_CONNECTOR_TYPE', default=EMPTY_STRING),
                "db_type":  env.str(x.upper() + '_SPARK_DB_TYPE', default=EMPTY_STRING),
                "hostname":  env.str(x.upper() + '_SPARK_HOSTNAME', default=EMPTY_STRING),
                "port":  env.int(x.upper() + '_SPARK_PORT', default=ZERO),
                "username":  env.str(x.upper() + '_SPARK_DATABASE_USERNAME', default=EMPTY_STRING),
                "password": env.str(x.upper() + '_SPARK_DATABASE_PASSWORD', default=EMPTY_STRING),
                "database_name": env.str(x.upper() + '_SPARK_DATABASE_NAME', default=EMPTY_STRING),
                "database_options": env.str(x.upper() + '_SPARK_DATABASE_OPTIONS', default=EMPTY_STRING)
            }
        },
        "hive": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_SPARK', default=False),
            "type": "spark",
            "handler": HiveHandler,
            "settings": {
                "jarpath": env.str(x.upper() + '_HIVE_JAR_PATH', default=EMPTY_STRING),
                "url": env.str(x.upper() + '_HIVE_URL', default=EMPTY_STRING),
                "uid": env.str(x.upper() + '_HIVE_UID', default=EMPTY_STRING),
                "pwd": env.str(x.upper() + '_HIVE_PWD', default=EMPTY_STRING)
            }
        },
        "livy": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_SPARK', default=False),
            "type": "spark",
            "handler": LivyHandler,
            "settings": {
                "url": env.str(x.upper() + '_LIVY_URL', default=EMPTY_STRING),
                "username": env.str(x.upper() + '_LIVY_USERNAME', default=EMPTY_STRING),
                "password": env.str(x.upper() + '_LIVY_PASSWORD', default=EMPTY_STRING),
                "driver_memory": env.str(x.upper() + '_LIVY_DRIVER_MEMORY', default="512m"),
                "driver_cores": env.int(x.upper() + '_LIVY_DRIVER_CORES', default=1),
                "executor_cores": env.int(x.upper() + '_LIVY_EXECUTOR_CORES', default=1),
                "executor_memory": env.str(x.upper() + '_LIVY_EXECUTOR_MEMORY', default="512m"),
                "num_executors": env.int(x.upper() + '_LIVY_NUM_EXECUTORS', default=1),
                "queue": env.str(x.upper() + '_LIVY_QUEUE', default="default"),
                "name": env.str(x.upper() + '_LIVY_NAME', default="mlapp"),
                "heartbeat_timeout": env.int(x.upper() + '_LIVY_HEARTBEAT_TIMEOUT', default=60)
            }
        },
        # default rabbitmq settings for Kubernetes deployment. See ML App Wiki documentation
        "rabbitmq": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_MQ', default=False),
            "type": "message_queue",
            "handler": RabbitMQHandler,
            "settings": {
                'hostname': env.str(x.upper() + '_RABBITMQ_HOSTNAME', default=EMPTY_STRING),
                "port": env.int(x.upper() + '_RABBITMQ_PORT', default=ZERO),
                'username': env.str(x.upper() + '_RABBITMQ_USERNAME', default=EMPTY_STRING),
                'password': env.str(x.upper() + '_RABBITMQ_PASSWORD', default=EMPTY_STRING),
                'cert_path': env.str(x.upper() + '_RABBITMQ_CERT_PATH', default=EMPTY_STRING),
                'connection_timeout': env.int(x.upper() + '_RABBITMQ_CONNECTION_TIMEOUT', default=15)  # seconds
            }
        },
        "boto": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_FS', default=False),
            "handler": BotoStorageHandler,
            "type": "file_storage",
            "settings": {
                'endpoint': env.str(x.upper() + '_BOTO_ENDPOINT', default=EMPTY_STRING),
                'accessKeyId': env.str(x.upper() + '_BOTO_ACCESS_KEY', default=EMPTY_STRING),
                'secretKey': env.str(x.upper() + '_BOTO_SECRET_KEY', default=EMPTY_STRING),
                'is_secure': env.bool(x.upper() + '_BOTO_SECURE', default=False)
            }
        },
        "ibm_boto3": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_FS', default=False),
            "handler": IBMBoto3StorageHandler,
            "type": "file_storage",
            "settings": {
                'endpoint': env.str(x.upper() + '_IBM_BOTO3_ENDPOINT', default=EMPTY_STRING),
                'api_key_id': env.str(x.upper() + '_IBM_BOTO3_API_KEY_ID', default=EMPTY_STRING),
                'service_crn': env.str(x.upper() + '_IBM_BOTO3_SERVICE_CRN', default=EMPTY_STRING)
            }
        },
        "kafka": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_MQ', default=False),
            "handler": KafkaHandler,
            "type": "message_queue",
            "settings": {
                'hostname': env.str(x.upper() + '_KAFKA_HOSTNAME', default=EMPTY_STRING),
                "port": env.int(x.upper() + '_KAFKA_PORT', default=ZERO),
                # 'listen_queues_names': env.list(x.upper() + '_KAFKA_LISTEN_QUEUES_NAMES', default=[]),
                # 'send_queue_names': env.list(x.upper() + '_KAFKA_SEND_QUEUE_NAMES', default=[]),
                'connection_timeout': env.int(x.upper() + '_KAFKA_CONNECTION_TIMEOUT', default=15),  # seconds
                "username": env.str(x.upper() + '_KAFKA_USERNAME', default=EMPTY_STRING),
                "password": env.str(x.upper() + '_KAFKA_PASSWORD', default=EMPTY_STRING),
            }
        },
        "kafka_kerberos": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_MQ', default=False),
            "handler": KafkaKerberosHandler,
            "type": "message_queue",
            "settings": {
                "hostnames": env.str(x.upper() + '_KAFKA_HOSTNAMES', default=EMPTY_STRING),
                "group_id": env.str(x.upper() + '_KAFKA_GROUP_ID', default=EMPTY_STRING),
                "keytab": env.str(x.upper() + '_KAFKA_KEYTAB', default=EMPTY_STRING),
                "ca_location": env.str(x.upper() + '_KAFKA_CA_LOCATION', default=EMPTY_STRING),
                "username": env.str(x.upper() + '_KAFKA_USERNAME', default=EMPTY_STRING),
                "domain_name": env.str(x.upper() + '_KAFKA_DOMAIN_NAME', default=EMPTY_STRING)
            }
        },
        "azureml_queue": lambda x: {
            "main": env.bool(x.upper() + '_MAIN_MQ', default=False),
            "handler": AMLQueue,
            "type": "message_queue",
            "settings": {
                'experiment_name': env.str(x.upper() + '_AML_EXPERIMENT_NAME', default=EMPTY_STRING),
            }
        }
    }


settings = {
    # assets output is saved locally at the path below.
    "local_storage_path": "output",
    "latest_file_name": "latest_ids.json",

    # assets output is fallback at the path below.
    "temporary_storage_path": "temporary_output",

    # if file storage configurations are set, pipeline will save output there instead of locally
    "file_store_buckets": {
        'objects': 'mlapp-objects',
        'csvs': 'mlapp-csvs',
        'configs': 'mlapp-configs',
        'metadata': 'mlapp-metadata',
        'imgs': 'mlapp-imgs',
        'logs': 'mlapp-logs'
    },

    # main queues for app to send/listen
    "queues": {
        'listen_queue_names': ['analysis_general_listen'],
        'send_queue_name': 'analysis_respond',
        'analysis_logs_queue_name': 'analysis_logs',
    },
    "pipelines": {
        'train': ['load_train_data', 'clean_train_data', 'transform_train_data', 'train_model'],
        'explore_data': ['load_train_data', 'clean_train_data', 'transform_train_data', 'visualization'],
        'feature_engineering': ['load_train_data', 'clean_train_data', 'transform_train_data', 'cache_features'],
        'reuse_features_and_train':['load_features','train_model'],
        'forecast': ['load_forecast_data', 'clean_forecast_data', 'transform_forecast_data', 'forecast'],
        'predictions_accuracy': ['load_actuals_data', 'update_actuals', 'evaluate_prediction_accuracy'],
        'retrain': ['load_train_data', 'clean_train_data', 'transform_train_data', 'train_model'],
        'train_flow': ['load_train_data', 'clean_train_data', 'transform_train_data', 'cache_features', 'train_model'],
        'forecast_flow': ['load_forecast_data', 'clean_forecast_data', 'transform_forecast_data', 'forecast']
    },
    # dynamically configured
    "services": {}
}
