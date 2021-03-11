import logging
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, Column, Time, Integer, String, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from mlapp.handlers.databases.database_interface import DatabaseInterface


class Job(declarative_base()):
    __tablename__ = 'jobs'

    id = Column(String, primary_key=True)
    user = Column(String)
    data = Column(JSON)
    status_code = Column(Integer)
    status_msg = Column(String)
    created_at = Column(Time)
    updated_at = Column(Time)


class SQLAlchemyHandler(DatabaseInterface):
    def __init__(self, settings):
        """
        Initializes the SQLAlchemyHandler
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        self.conn = None
        self.engine = None
        self.connections_parameters = settings
        self.char = '%s'
        if self.connections_parameters == {}:
            logging.error('Missing connection parameters!')

    def _connect_database(self):
        """
        Connect to the database.
        """
        try:
            self.engine = self._create_engine()
            self.conn = self.engine.connect()
        except Exception as e:
            raise e

    def _create_engine(self):
        """
        Create engine for the database.
        """
        if hasattr(self, 'connection_string'):
            return create_engine(self.connection_string, connect_args=self.connections_parameters['options'])
        else:
            raise NotImplementedError("Must initiate `self.connection_string` in your handler's `__init__` function!")

    def _close_connection(self):
        """
        Close connection to the database and dispose of engine.
        """
        try:
            if self.conn is not None and not self.conn.closed:
                self.conn.close()
                self.engine.dispose()
        except Exception as e:
            raise e

    def execute_query(self, query, params=None):
        """
        Executes Query in the database.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query
        """
        self._connect_database()
        try:
            query, params = self._format_params_and_query(query, params)
            if params is None:
                params = []

            result = self.conn.execute(query, tuple(params))  # Syntax error in query
            self.conn.close()
            self.engine.dispose()
            if result.returns_rows:
                # has returned rows
                return list(result)
            else:
                # number of rows matched by where criterion of an UPDATE or DELETE
                return result.rowcount
        except Exception as e:
            self._close_connection()
            raise e

    def insert_query(self, query, values):
        """
        Executes an "INSERT" query in the database.
        :param query: "insert into table_name (column1, comlum2,,,)VALUES (%s,%s,,,)
        :param values: list of dictionaries i.e. [{value1,value2,,,}, {value11,value22,,,,},,,]
        :return: Void
        """
        try:
            self._connect_database()
            self.conn.execute(query, values)
            self.conn.close()
            self.engine.dispose()
        except Exception as e:
            self._close_connection()
            raise e

    def insert_df(self, sql_table, df, batch_length=1000):
        """
        Inserts a DataFrame into a table in the database.
        :param sql_table: str - name of the table.
        :param df: DataFrame (Pandas, PySpark or other) - Matrix type DataFrame containing all values to insert.
        :param batch_length: int - length of the how many rows to insert from matrix at a time
        """
        try:
            num_batches = np.ceil(np.true_divide(len(df.index), batch_length))
            print('>>>>>>  Insertion of {0} batches  >>>>  '.format(str(num_batches)))
            for i in range(int(num_batches)):
                self._connect_database()
                try:
                    cur_df = df.iloc[i * batch_length: min((i + 1) * batch_length, len(df.index))]
                    cur_df.to_sql(sql_table, con=self.conn, if_exists='append', index=False)
                except Exception as e:
                    print(e)
                    self.conn.close()
                    self.engine.dispose()
                    continue
                self.conn.close()
                self.engine.dispose()
        except Exception as e:
            self._close_connection()
            raise e

    def get_df(self, query, params=None):
        """
        Executes a query in the database and returns it as a DataFrame.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query as a DataFrame
        """
        self._connect_database()
        try:
            query, params = self._format_params_and_query(query, params)
            if params is None:
                params = []
            result = self.conn.execute(query, tuple(params))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            self.conn.close()
            self.engine.dispose()
            return df
        except Exception as e:
            self._close_connection()
            raise e

    def update_job_running(self, job_id):
        """
        Updates row in the table of jobs by the job_id to status `Running`
        Functionality of the MLCP (Machine Learning Control Panel)
        :param job_id: str - id of the job
        """
        self._connect_database()
        try:
            # create session
            session = sessionmaker(bind=self.engine)()

            # query by job id
            q = session.query(Job)
            q = q.filter(Job.id == job_id)

            # create or update
            records = q.all()
            if records:
                records[0].status_code = 1
                records[0].status_msg = 'running'
            else:
                session.add(Job(id=job_id, status_code=1, status_msg='running', user='DefaultUser',
                                created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # commit and close
            session.commit()
            session.close()
            self._close_connection()
        except Exception as e:
            self._close_connection()
            raise e

    def update_actuals(self, df, index_column='index',target_column='y_true'):
        # TODO: update to ORM (currently supports postgres only)
        query = 'update target set y_true = (case'
        columns = list(df)

        for index in columns:
            value = float(df[index])
            query += f" when index = '{str(index)}' then {str(value)}"
        query += ' end) where type=3 and index in (' + ','.join([f"'{x}'" for x in columns ]) + ')'
        self.execute_query(query)

    def get_model_predictions(self, model_id, prediction_type=3, from_date=None, to_date=None):
        # TODO: update to ORM (currently supports postgres only)
        # TODO: use from and to dates
        return self.get_df(
            query="select * from target where model_id = ? and type = ? ",
            params=[model_id, prediction_type]
        )

    def _format_params_and_query(self, query, params):
        """
        Formatting query and parameters. Replaces {number} with self.char (%s or other).
        Also knows to handle list parameters and split them into single parameters.
        :param query: str - query to be executed.
        :param params: list - list of parameters to be used if necessary in query
        :return: result of query and new parameters
        """
        if query.__contains__('{') and query.__contains__('}'):
            new_params = []
            # we need to replace the str.format() placeholders with %s
            counter = 0
            for param in params:
                new_placeholder = self.char
                old_placeholder = '{' + str(counter) + '}'

                if query.__contains__('"' + old_placeholder + '"'):
                    old_placeholder = '"' + old_placeholder + '"'

                if isinstance(param, list) or isinstance(param, tuple):
                    new_params.extend(param)
                    new_placeholder = ', '.join([self.char] * len(param))

                    if not query.__contains__('(' + new_placeholder) or not query.__contains__('( ' + new_placeholder):
                        new_placeholder = '(' + new_placeholder + ')'
                else:
                    new_params.extend([param])
                query = query.replace(old_placeholder, new_placeholder)
                counter += 1
            return query, new_params
        else:
            return query, params

    def __del__(self):
        self._close_connection()
