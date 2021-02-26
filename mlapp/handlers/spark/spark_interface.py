from abc import ABCMeta, abstractmethod


class SparkInterface:
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_model(self, file_path, module):
        """
        Loads a spark model
        :param file_path: path to spark model file
        :param module: name of module to load
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def exec_query(self, query, params=None, **kwargs):
        """
        Executes Query in the database.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

