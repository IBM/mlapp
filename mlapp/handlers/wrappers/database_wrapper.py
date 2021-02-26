from mlapp.handlers.wrappers.wrapper_interface import WrapperInterface


class DatabaseWrapper(WrapperInterface):
    def init(self):
        """
        Initializes the Wrapper for all handlers of `database` type.
        """
        super(DatabaseWrapper, self).init('database')

    def execute_query(self, query, params=None):
        """
        Executes Query in the database.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query
        """
        for handler_name in self._main_handlers:
            result = self._handlers[handler_name].execute_query(query, params)
            if hasattr(result, '__len__'):
                return result

    def insert_query(self, query, values):
        """
        Executes an "INSERT" query in the database.
        :param query: str - query to be executed.
        :param values: list - list of values to be used in the query
        """
        for handler_name in self._main_handlers:
            self._handlers[handler_name].insert_query(query, values)

    def insert_df(self, sql_table, df, batch_length=1000):
        """
        Inserts a DataFrame into a table in the database.
        :param sql_table: str - name of the table.
        :param df: DataFrame (Pandas, PySpark or other) - Matrix type DataFrame containing all values to insert.
        :param batch_length: int - length of the how many rows to insert from matrix at a time
        """
        for handler_name in self._main_handlers:
            self._handlers[handler_name].insert_df(sql_table, df, batch_length)

    def get_df(self, query, params=None):
        """
        Executes a query in the database and returns it as a DataFrame.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query as a DataFrame
        """
        for handler_name in self._main_handlers:
            return self._handlers[handler_name].get_df(query, params)

    def update_job_running(self, job_id):
        """
        Updates row in the table of jobs by the job_id to status `Running`
        Functionality of the MLCP (Machine Learning Control Panel)
        :param job_id: str - id of the job
        """
        for handler_name in self._main_handlers:
            self._handlers[handler_name].update_job_running(job_id)

    def update_actuals(self, df):
        """
        Update target table with the y_true
        :param df: the dataframe that represents the real data that was loaded
        :return: none
        """
        for handler_name in self._main_handlers:
            self._handlers[handler_name].update_actuals(df)

    def get_model_predictions(self, model_id, prediction_type=3, from_date=None, to_date=None):
        """
        Update target table with the y_true
        :param df: the dataframe that represents the real data that was loaded
        :return: none
        """
        for handler_name in self._main_handlers:
            return self._handlers[handler_name].get_model_predictions(
                model_id, prediction_type=3, from_date=None, to_date=None
            )


database_instance = DatabaseWrapper()
