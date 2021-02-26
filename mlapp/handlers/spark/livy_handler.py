from mlapp.handlers.spark.spark_interface import SparkInterface
from livy import LivySession
from requests.auth import HTTPBasicAuth
from textwrap import dedent
import requests


class LivyHandler(SparkInterface):

    def __init__(self, settings):
        """
        Initializes the HiveHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        settings = {key: value for (key, value) in settings.items() if not value == ""}

        self.livy_settings = settings

        requests_session = requests.Session()
        requests_session.headers.update({'X-Requested-By': 'livy', 'Content-Type': 'application/json'})

        self.params = {
            "url": self.livy_settings.get('url'),
            "auth": HTTPBasicAuth(self.livy_settings.get('username'), self.livy_settings.get('password')),
            "driver_memory": self.livy_settings.get('driver_memory', "512m"),
            "driver_cores": self.livy_settings.get('driver_cores', 1),
            "executor_cores": self.livy_settings.get('executor_cores', 1),
            "executor_memory": self.livy_settings.get('executor_memory', "512m"),
            "num_executors": self.livy_settings.get('num_executors', 1),
            "queue": self.livy_settings.get('queue', "default"),
            "name": self.livy_settings.get('name', "mlapp"),
            "heartbeat_timeout": self.livy_settings.get('heartbeat_timeout', 60),
            "requests_session": requests_session
        }

    def exec_query(self, query, params=None, **kwargs):
        """
        Executes Query in the database.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query
        """
        df_variable_name = kwargs.get('df_variable_name', 'df')

        with LivySession.create(**self.params) as session:
            # Run some code on the remote cluster
            session.run(dedent("""
                spark.sql('use default')
                df = spark.sql('show databases')
            """.format(*params)))

            # Retrieve the result
            df = session.read(df_variable_name)

        return df
