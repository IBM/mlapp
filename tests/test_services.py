import unittest
import warnings


class TestServices(unittest.TestCase):
    # databases
    def test_mysql(self):
        try:
            from mlapp.handlers.databases.mysql_handler import MySQLHandler
        except ImportError:
            warnings.warn("Missing `PyMySQL` installation for the test.")

    def test_mssql(self):
        try:
            from mlapp.handlers.databases.mssql_handler import MssqlHandler
        except ImportError:
            warnings.warn("Missing `pyodbc` installation for the test.")

    def test_postgres(self):
        try:
            from mlapp.handlers.databases.postgres_handler import PostgresHandler
        except ImportError:
            warnings.warn("Missing `pg8000<=1.16.5` installation for the test.")

    def test_snowflake(self):
        try:
            from mlapp.handlers.databases.snowflake_handler import SnowflakeHandler
        except ImportError:
            warnings.warn("Missing `snowflake-sqlalchemy` installation for the test.")

    def test_aml_postgres(self):
        try:
            from mlapp.handlers.file_storages.aml_run_storage_handler import AmlRunStorageHandler
        except ImportError:
            warnings.warn("Missing `azureml-sdk` and/or `pq8000` installation for the test.")

    def test_aml_metadata(self):
        try:
            from mlapp.handlers.file_storages.aml_model_storage_handler import AmlModelStorageHandler
        except ImportError:
            warnings.warn("Missing `azureml-sdk` installation for the test.")

    # file storages
    def test_minio(self):
        try:
            from mlapp.handlers.file_storages.minio_storage_handler import MinioStorageHandler
        except ImportError:
            warnings.warn("Missing `minio` installation for the test.")

    def test_azure_blob(self):
        try:
            from mlapp.handlers.file_storages.azure_blob_handler import AzureBlobHandler
        except ImportError:
            warnings.warn("Missing `azure-storage-blob` installation for the test.")

    def test_boto3(self):
        try:
            from mlapp.handlers.file_storages.boto_storage_handler import BotoStorageHandler
        except ImportError:
            warnings.warn("Missing `boto3` installation for the test.")

    def test_ibm_boto3(self):
        try:
            from mlapp.handlers.file_storages.ibm_boto3_storage_handler import IBMBoto3StorageHandler
        except ImportError:
            warnings.warn("Missing `ibm-cos-sdk` installation for the test.")

    # message queues
    def test_rabbitmq(self):
        try:
            from mlapp.handlers.message_queues.rabbitmq_handler import RabbitMQHandler
        except ImportError:
            warnings.warn("Missing `pika` installation for the test.")

    def test_aml_queue(self):
        try:
            from mlapp.handlers.message_queues.aml_queue import AMLQueue
        except ImportError:
            warnings.warn("Missing `azureml-sdk` installation for the test.")

    def test_azure_servicebus(self):
        try:
            from mlapp.handlers.message_queues.azure_service_bus_handler import AzureServicesBusHandler
        except ImportError:
            warnings.warn("Missing `azure-servicebus` installation for the test.")

    def test_kafka(self):
        try:
            from mlapp.handlers.message_queues.kafka_handler import KafkaHandler
        except ImportError:
            warnings.warn("Missing `kafka-python` installation for the test.")

    def test_kafka_kerberos(self):
        try:
            from mlapp.handlers.message_queues.kafka_kerberos_handler import KafkaKerberosHandler
        except ImportError as e:
            warnings.warn("Missing `confluent_kafka` installation for the test.")

    # spark
    def test_spark(self):
        try:
            from mlapp.handlers.spark.spark_handler import SparkHandler
        except ImportError:
            warnings.warn("Missing `pyspark` installation for the test.")

    def test_livy(self):
        try:
            from mlapp.handlers.spark.livy_handler import LivyHandler
        except ImportError:
            warnings.warn("Missing `livy` installation for the test.")


if __name__ == '__main__':
    unittest.main()
