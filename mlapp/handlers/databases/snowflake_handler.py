from mlapp.handlers.databases.sql_alchemy_handler import SQLAlchemyHandler


class SnowflakeHandler(SQLAlchemyHandler):
    def __init__(self, settings):
        """
        Initializes the ￿SnowflakeHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(SnowflakeHandler, self).__init__(settings)

        authenticator = '&authenticator=externalbrowser' if self.connections_parameters['password'] == '' else ''
        self.connection_string = 'snowflake://{user}:{password}@{account}/{database}/{schema}?' \
                                 'warehouse={warehouse}&role={role}{authenticator}'\
            .format(user=self.connections_parameters['user'],
                    password=self.connections_parameters['password'],
                    account=self.connections_parameters['account'],
                    database=self.connections_parameters['database'],
                    schema=self.connections_parameters['schema'],
                    warehouse=self.connections_parameters['warehouse'],
                    role=self.connections_parameters['role'],
                    authenticator=authenticator)
