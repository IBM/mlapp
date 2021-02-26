from mlapp.mlapp_cli.common.cli_utilities import to_lower, clean_spaces, is_int, to_int, list_one_check

add_services_options = {
    "mysql": {
        "MLAPP_SERVICE_TYPE": 'mysql',
        "MAIN_DB": {
            "display_name": "Is it your main database",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        "MYSQL_HOSTNAME": {
            "display_name": "Hostname",
            "short_description": "Press enter to set hostname as localhost",
            "default": '127.0.0.1',
            "required": True
        },
        "MYSQL_PORT": {
            "display_name": "Port",
            "short_description": 'Press enter to set port to 3306',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Port should contain digits only.",
            "default": 3306,
            "required": True
        },
        "MYSQL_DATABASE_NAME": {
            "display_name": "Database name",
            "short_description": 'Enter your database name',
            "required": True
        },
        "MYSQL_USER_ID": {
            "display_name": "Username",
            "short_description": 'Enter your database user id',
            "required": True
        },
        "MYSQL_PASSWORD": {
            "display_name": "Password",
            "short_description": 'Enter your database password'
        }
    },
    "postgres": {
        "MLAPP_SERVICE_TYPE": 'postgres',
        "MAIN_DB": {
            "display_name": "Is it your main database",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        "POSTGRES_HOSTNAME": {
            "display_name": "Hostname",
            "short_description": "Press enter to set hostname as localhost",
            "default": '127.0.0.1',
            "required": True
        },
        "POSTGRES_PORT": {
            "display_name": "Port",
            "short_description": 'Press enter to set port to 5432',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Port should contain digits only.",
            "default": 5432,
            "required": True
        },
        "POSTGRES_DATABASE_NAME": {
            "display_name": "Database name",
            "short_description": 'Enter your database name',
            "required": True
        },
        "POSTGRES_USER_ID": {
            "display_name": "Username",
            "short_description": 'Enter your database user id',
            "required": True
        },
        "POSTGRES_PASSWORD": {
            "display_name": "Password",
            "short_description": 'Enter your database password'
        }
    },
    "mssql": {
        "MLAPP_SERVICE_TYPE": 'mssql',
        "MAIN_DB": {
            "display_name": "Is it your main database",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        "MSSQL_HOSTNAME": {
            "display_name": "Hostname",
            "short_description": "Press enter to set hostname as localhost",
            "default": '127.0.0.1',
            "required": True
        },
        "MSSQL_PORT": {
            "display_name": "Port",
            "short_description": 'Press enter to set port to 1433',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Port should contain digits only.",
            "default": 1433,
            "required": True
        },
        "MSSQL_DATABASE_NAME": {
            "display_name": "Database name",
            "short_description": 'Enter your database name',
            "required": True
        },
        "MSSQL_USER_ID": {
            "display_name": "Username",
            "short_description": 'Enter your database user id',
            "required": True
        },
        "MSSQL_PASSWORD": {
            "display_name": "Password",
            "short_description": 'Enter your database password'
        }
    },
    "snowflake": {
        "MLAPP_SERVICE_TYPE": 'snowflake',
        "MAIN_DB": {
            "display_name": "Is it your main database",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        "SNOWFLAKE_ACCOUNT": {
            "display_name": "Account",
            "short_description": "Enter your account, i.e:  xxx.east-us-2.azure",
            "required": True
        },
        "SNOWFLAKE_USER": {
            "display_name": "Username",
            "short_description": 'Enter your database user id',
            "required": True
        },
        "SNOWFLAKE_PASSWORD": {
            "display_name": "Password (Press `Enter` to keep empty in case you're using an active directory)",
            "short_description": 'Enter your database password'
        },
        "SNOWFLAKE_DATABASE": {
            "display_name": "Database",
            "short_description": 'Enter your database name'
        },
        "SNOWFLAKE_SCHEMA": {
            "display_name": "Schema",
            "short_description": 'Enter your schema name'
        },
        "SNOWFLAKE_WAREHOUSE": {
            "display_name": "Warehouse",
            "short_description": 'Enter your warehouse name'
        },
        "SNOWFLAKE_ROLE": {
            "display_name": "Role",
            "short_description": 'Enter your role'
        }
    },
    "rabbitmq": {
        "MLAPP_SERVICE_TYPE": 'rabbitmq',
        "MAIN_MQ": {
            "display_name": "Is it your main message queue",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        'RABBITMQ_HOSTNAME': {
            "display_name": "Hostname",
            "short_description": "Press enter to set hostname as localhost",
            "default": 'amqp://127.0.0.1',
            "required": True
        },
        'RABBITMQ_PORT': {
            "display_name": "Port",
            "short_description": 'Press enter to set port to 5673',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Port should contain digits only.",
            "default": 5673,
            "required": True
        },
        # 'RABBITMQ_LISTEN_QUEUES_NAMES': {
        #     "display_name": "Listen queues names",
        #     "short_description": 'Press enter listening queues names, separated by comma',
        #     "transformations": [clean_spaces, list_one_check],
        #     "required": True
        # },
        # 'RABBITMQ_SEND_QUEUE_NAMES': {
        #     "display_name": "Send queues names",
        #     "short_description": 'Press enter sending queues names, separated by comma',
        #     "transformations": [clean_spaces, list_one_check],
        #     "required": True
        # },
        'RABBITMQ_CONNECTION_TIMEOUT': {
            "display_name": "Connection timeout",
            "short_description": 'Enter your connection timeout number in seconds, default 15',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Connection timeout should contain digits only.",
            "default": 15,
            "required": True
        }
    },
    "minio": {
        "MLAPP_SERVICE_TYPE": 'minio',
        "MAIN_FS": {
            "display_name": "Is it your main file storage",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        'MINIO_ENDPOINT': {
            "display_name": "Endpoint",
            "short_description": "Press enter to set endpoint as localhost",
            "default": '127.0.0.1',
            "required": True
        },
        'MINIO_ACCESS_KEY': {
            "display_name": "Access key",
            "short_description": "Enter your access key token",
            "required": True
        },
        'MINIO_SECRET_KEY': {
            "display_name": "Secret key",
            "short_description": "Enter your secret key token",
            "required": True
        },
        'MINIO_PORT': {
            "display_name": "Port",
            "short_description": 'Press enter to set port to 9000',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Port should contain digits only.",
            "default": 9000,
            "required": True

        },
        'MINIO_SECURE': {
            "display_name": "Secure",
            "short_description": 'Secure every bucket, Y/N, default no',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'false',
            "required": False

        },
        'MINIO_REGION': {
            "display_name": "Region",
            "short_description": "Enter your cluster region (Optional)",
            "required": False
        }
    },
    "azure-blob": {
        "MLAPP_SERVICE_TYPE": 'azure_blob',
        "MAIN_FS": {
            "display_name": "Is it your main file storage",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        'AZURE_BLOB_ACCOUNT_NAME': {
            "display_name": "account name",
            "short_description": "Enter your account name token",
            "required": True
        },
        'AZURE_BLOB_ACCOUNT_KEY': {
            "display_name": "account key",
            "short_description": "Enter your account key",
            "required": True
        }
    },
    "databricks": {
        "MLAPP_SERVICE_TYPE": 'databricks',
        "MAIN_SPARK": {
            "display_name": "Is it your main spark cluster",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        'DATABRICKS_ADDRESS': {
            "display_name": "Address",
            "short_description": "Press enter to set address as localhost",
            "default": '127.0.0.1',
            "required": True
        },
        'DATABRICKS_TOKEN': {
            "display_name": "Token",
            "short_description": "Enter your Databricks cluster authorization token",
            "required": True
        },
        'DATABRICKS_CLUSTER_ID': {
            "display_name": "Cluster ID",
            "short_description": 'Enter your Databricks cluster id',
            "required": True
        },
        'DATABRICKS_PORT': {
            "display_name": "Port",
            "short_description": 'Press enter to set port to 15001',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Port should contain digits only.",
            "default": 15001,
            "required": True
        },
        'DATABRICKS_ORGANIZATION_ID': {
            "display_name": "Organization ID",
            "short_description": 'Enter your Databricks organization id',
            "required": True
        }
    },
    "azure-service-bus": {
        "MLAPP_SERVICE_TYPE": 'azure_service_bus',
        "MAIN_MQ": {
            "display_name": "Is it your main message queue",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        'AZURE_SERVICE_BUS_HOSTNAME': {
            "display_name": "Host Name",
            "short_description": "Enter the Host Name",
            "required": True
        },
        'AZURE_SERVICE_BUS_SHARED_ACCESS_KEY_NAME': {
            "display_name": "Shared Access Key Name (Policy Name)",
            "short_description": "Enter the Policy Name",
            "required": True
        },
        'AZURE_SERVICE_BUS_SHARED_ACCESS_KEY': {
            "display_name": "Shared Access Key (Primary Key)",
            "short_description": "Enter the Primary Key",
            "required": True
        }
    },
    "kafka": {
        "MLAPP_SERVICE_TYPE": 'kafka',
        "MAIN_MQ": {
            "display_name": "Is it your main message queue",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        'KAFKA_HOSTNAME': {
            "display_name": "Hostname",
            "short_description": "Press enter to set hostname as localhost",
            "default": '127.0.0.1',
            "required": True
        },
        'KAFKA_PORT': {
            "display_name": "Port",
            "short_description": 'Press enter to set port to 9092',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Port should contain digits only.",
            "default": 9092,
            "required": True
        },
        'KAFKA_CONNECTION_TIMEOUT': {
            "display_name": "Connection timeout",
            "short_description": 'Enter your connection timeout number in seconds, default 15',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Connection timeout should contain digits only.",
            "default": 15,
            "required": True
        }
    },
    "boto": {
        "MLAPP_SERVICE_TYPE": 'boto',
        "MAIN_FS": {
            "display_name": "Is it your main file storage",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        'BOTO_ACCESS_KEY': {
            "display_name": "Access key",
            "short_description": "Enter your access key token",
            "required": True
        },
        'BOTO_SECRET_KEY': {
            "display_name": "Secret key",
            "short_description": "Enter your secret key token",
            "required": True
        }
    },
    "spark-local": {
        "MLAPP_SERVICE_TYPE": 'spark_local',
        "MAIN_SPARK": {
            "display_name": "Is it your main local spark cluster",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        }
    },
    "spark": {
        "MLAPP_SERVICE_TYPE": 'spark',
        "MAIN_SPARK": {
            "display_name": "Is it your main spark cluster",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        }
    },
    "azureml-model-storage": {
        "MLAPP_SERVICE_TYPE": 'azureml_model_storage',
        "MAIN_FS": {
            "display_name": "Is it your main azureml file storage",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        }
    },
    "azureml-run-storage": {
        "MLAPP_SERVICE_TYPE": 'azureml_run_storage',
        "MAIN_FS": {
            "display_name": "Is it your main azureml file storage",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        "AML_RUN_STORAGE_HOSTNAME": {
            "display_name": "Hostname",
            "short_description": "Press enter to set hostname as localhost",
            "default": '127.0.0.1',
            "required": True
        },
        "AML_RUN_STORAGE_PORT": {
            "display_name": "Port",
            "short_description": 'Press enter to set port to 5432',
            "validations": [is_int],
            "transformations": [to_int],
            "error_msg": "Port should contain digits only.",
            "default": 5432,
            "required": True
        },
        "AML_RUN_STORAGE_DATABASE_NAME": {
            "display_name": "Database name",
            "short_description": 'Enter your database name',
            "required": True
        },
        "AML_RUN_STORAGE_USER_ID": {
            "display_name": "Username",
            "short_description": 'Enter your database user id',
            "required": True
        },
        "AML_RUN_STORAGE_PASSWORD": {
            "display_name": "Password",
            "short_description": 'Enter your database password'
        }
    },
    "azureml-queue": {
        "MLAPP_SERVICE_TYPE": 'azureml_queue',
        "MAIN_MQ": {
            "display_name": "Is it your main message queue",
            "short_description": 'Y/N, default yes',
            "transformations": [to_lower, clean_spaces],
            "values": {
                'y': 'true',
                'n': 'false',
                'yes': 'true',
                'no': 'false'
            },
            "error_msg": "Possible values should be 'y', 'n', 'yes' or 'no'.",
            "default": 'true',
            "required": True
        },
        'AML_EXPERIMENT_NAME': {
            "display_name": "experiment name",
            "required": False
        },
    }
}
