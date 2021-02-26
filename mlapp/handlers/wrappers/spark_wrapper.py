from mlapp.handlers.wrappers.wrapper_interface import WrapperInterface


class SparkWrapper(WrapperInterface):
    def init(self):
        """
        Initializes the Wrapper for all handlers of `spark` type.
        """
        super(SparkWrapper, self).init('spark')

    def load_model(self, file_path, module):
        """
        Loads a spark model
        :param file_path: path to spark model file
        :param module: name of module to load
        """
        for handler_name in self._main_handlers:
            return self._handlers[handler_name].load_model(file_path, module)

    def exec_query(self, query, params=None):
        """
        Executes Query in the database.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query
        """
        for handler_name in self._main_handlers:
            self._handlers[handler_name].exec_query(query, params)

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
        for handler_name in self._main_handlers:
            return self._handlers[handler_name].exec_query(
                file_path, sep=sep, header=header, toPandas=toPandas, **kwargs)


spark_instance = SparkWrapper()


