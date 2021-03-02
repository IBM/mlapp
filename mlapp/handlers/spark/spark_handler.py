from pyspark.sql import SparkSession, SQLContext, DataFrame
from pyspark import SparkConf
from mlapp.handlers.spark.spark_interface import SparkInterface


class SparkHandler(SparkInterface):

    def __init__(self, settings):
        """
        Initializes the SparkHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        conf = SparkConf()
        settings = {key: value for (key, value) in settings.items() if not value == ""}
        conf.setAll(settings.items())
        if settings.get("enable_hive", False):
            self.spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
        else:
            self.spark = SparkSession.builder.config(conf=conf).getOrCreate()

        # Get all database configuration
        self.spark_settings = settings
        self.driver = self.spark_settings.get('driver')
        self.connector_type = "jdbc"
        self.db_type = self.spark_settings.get('db_type')
        self.hostname = self.spark_settings.get('hostname')
        self.port = self.spark_settings.get('port')
        self.username = self.spark_settings.get('username')
        self.password = self.spark_settings.get('password')
        self.database_name = self.spark_settings.get('database_name')
        self.database_options = self.spark_settings.get('database_options')
        self.url = None
        if self.driver is not None and self.connector_type is not None and self.db_type is not None and \
           self.hostname is not None and self.port is not None and self.username is not None and \
           self.password is not None and self.database_name is not None and self.database_options is not None:
            self.__generate_url__()

    # def __del__(self):
    #     # Handler destructor
    #     self.close_connection()
    #
    # def close_connection(self):
    #     """
    #     Close the connection to spark
    #     """
    #     try:
    #         if (self.spark is not None) and (self.spark.stop is not None):
    #             self.spark.stop()
    #     except Exception as e:
    #         raise e

    def load_csv_file(self, file_path, sep=',', header=True, toPandas=False, **kwargs):
        """
        This function reads a csv file and return a spark DataFrame
        :param file_path: path to csv file
        :param sep: separator of csv file
        :param header: include header of file
        :param toPandas: to load as pandas DataFrame
        :param kwargs: other keyword arguments containing additional information
        :return: spark DataFrame (or pandas)
        """
        try:
            df = self.spark.read.csv(path=file_path, header=header, sep=sep, **kwargs)
            df = df.toPandas() if toPandas else df

            return df
        except Exception as e:
            raise e

    def load_model(self, file_path, module):
        """
        Loads a spark model
        :param file_path: path to spark model file
        :param module: name of module to load
        """
        model_class_name = file_path.split('.')[1]
        try:
            exec(module)
            return eval(model_class_name).load(file_path)
        except Exception as e:
            print(str(e))
            raise Exception("Missing import of `{}` in `spark_db_handler.py`!".format(model_class_name + 'Model'))

    def exec_query(self, query, params=None, **kwargs):
        """
        Executes Query in the database.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query
        """
        if params is not None:
            query = query.format(*params)

        return self.spark.sql(query)

    def __generate_url__(self):
        """
        This method creates the database url with all options for connect to the database
        :return: void
        """

        semicolon = ":"
        double_slash = "//"
        single_slash = "/"
        self.url = self.connector_type + semicolon + self.db_type + semicolon + double_slash + \
            self.hostname + semicolon + \
            single_slash + self.database_name + self.database_options
