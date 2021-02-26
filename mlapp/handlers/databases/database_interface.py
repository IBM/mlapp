from abc import ABCMeta, abstractmethod


class DatabaseInterface:
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute_query(self, query, params=None):
        """
        Executes Query in the database.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query
        """
        raise NotImplementedError()

    @abstractmethod
    def insert_query(self, query, values):
        """
        Executes an "INSERT" query in the database.
        :param query: str - query to be executed.
        :param values: list - list of values to be used in the query
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def insert_df(self, sql_table, df, batch_length=1000):
        """
        Inserts a DataFrame into a table in the database.
        :param sql_table: str - name of the table.
        :param df: DataFrame (Pandas, PySpark or other) - Matrix type DataFrame containing all values to insert.
        :param batch_length: int - length of the how many rows to insert from matrix at a time
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def get_df(self, query, params=None):
        """
        Executes a query in the database and returns it as a DataFrame.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query as a DataFrame
        """
        raise NotImplementedError()

    @abstractmethod
    def update_job_running(self, job_id):
        """
        Updates row in the table of jobs by the job_id to status `Running`
        Functionality of the MLCP (Machine Learning Control Panel)
        :param job_id: str - id of the job
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def update_actuals(self, df):
        """
        Update target table with the y_true
        :param df: the dataframe that represents the real data that was loaded
        :return: none
        """
        raise NotImplementedError()
