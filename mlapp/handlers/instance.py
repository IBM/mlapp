from mlapp.handlers.databases.database_interface import DatabaseInterface
from mlapp.handlers.file_storages.file_storage_interface import FileStorageInterface
from mlapp.handlers.message_queues.message_queue_interface import MessageQueueInterface
from mlapp.handlers.spark.spark_interface import SparkInterface
from mlapp.handlers.wrappers.database_wrapper import database_instance
from mlapp.handlers.wrappers.file_storage_wrapper import file_storage_instance
from mlapp.handlers.wrappers.message_queue_wrapper import message_queue_instance
from mlapp.handlers.wrappers.spark_wrapper import spark_instance


def _get_handler(instance_type, handler_name):
    handler = instance_type.get(handler_name)
    if handler is not None:
        return handler
    raise Exception("Handler %s not found" % str(handler_name))


def database_handler(handler_name) -> DatabaseInterface:
    return _get_handler(database_instance, handler_name)


def file_storage_handler(handler_name) -> FileStorageInterface:
    return _get_handler(file_storage_instance, handler_name)


def message_queue_handler(handler_name) -> MessageQueueInterface:
    return _get_handler(message_queue_instance, handler_name)


def spark_handler(handler_name) -> SparkInterface:
    return _get_handler(spark_instance, handler_name)

