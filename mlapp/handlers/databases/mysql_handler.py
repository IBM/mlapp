from mlapp.handlers.databases.sql_alchemy_handler import SQLAlchemyHandler


class MySQLHandler(SQLAlchemyHandler):
    def __init__(self, settings):
        """
        Initializes the ￿MySQLHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(MySQLHandler, self).__init__(settings)
        self.connection_string = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
            self.connections_parameters['user_id'],
            self.connections_parameters['password'],
            self.connections_parameters['hostname'],
            str(self.connections_parameters['port']),
            self.connections_parameters['database_name'])

