from mlapp.handlers.databases.sql_alchemy_handler import SQLAlchemyHandler
import ssl


class PostgresHandler(SQLAlchemyHandler):
    def __init__(self, settings):
        """
        Initializes the ￿PostgresHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        # add ssl
        if settings.get('ssl'):
            ssl_context = ssl.SSLContext()
            # ssl_context.verify_mode = ssl.CERT_REQUIRED
            # ssl_context.load_verify_locations('./certs/postgres.pem')
            settings['options'] = {
                'ssl_context': ssl_context
            }

        super(PostgresHandler, self).__init__(settings)

        self.connection_string = 'postgresql+pg8000://{0}:{1}@{2}:{3}/{4}'.format(
            self.connections_parameters['user_id'],
            self.connections_parameters['password'],
            self.connections_parameters['hostname'],
            str(self.connections_parameters['port']),
            self.connections_parameters['database_name'])

