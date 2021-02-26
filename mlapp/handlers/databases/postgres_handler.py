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

    def update_actuals(self, df, index_column='index',target_column='y_true'):
        query = 'update target set y_true = (case'
        columns = list(df)

        for index in columns:
            value = float(df[index])
            query += f" when index = '{str(index)}' then {str(value)}"
        query += ' end) where type=3 and index in (' + ','.join([f"'{x}'" for x in columns ]) + ')'
        self.execute_query(query)

    def get_model_predictions(self, model_id, prediction_type=3, from_date=None, to_date=None):
        # TODO: add dates
        return self.get_df(
            query="select * from target where model_id = ? and type = ? ",
            params=[model_id, prediction_type]
        )
