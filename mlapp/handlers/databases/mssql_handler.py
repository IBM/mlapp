import urllib
from mlapp.handlers.databases.sql_alchemy_handler import SQLAlchemyHandler


class MssqlHandler(SQLAlchemyHandler):

    def __init__(self, settings):
        """
        Initializes the ￿MssqlHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(MssqlHandler, self).__init__(settings)

        # preparing connection string
        connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};' + \
                            'SERVER={0};DATABASE={1};UID={2};PWD={3};Port={4};'.format(
                                self.connections_parameters['hostname'],
                                self.connections_parameters['database_name'],
                                self.connections_parameters['user_id'],
                                self.connections_parameters['password'],
                                self.connections_parameters['port'])
        connection_string = urllib.parse.quote_plus(connection_string)
        self.char = '?'
        self.connection_string = "mssql+pyodbc:///?odbc_connect=%s" % connection_string
