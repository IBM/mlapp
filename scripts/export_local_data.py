from mlapp.main import MLApp
from mlapp.config import settings
import pandas as pd

if __name__ == "__main__":
    config = {
        'handler_name': 'handler',
        'files': [{
            'table_name': 'table',
            'file_name': 'file_name.csv'
        }]
    }

    mlapp = MLApp({'env_file_path': 'path/to/.env'})
    handlers = {}
    for service_name in settings.get('services', []):
        service_item = settings['services'][service_name]
        try:
            handlers[service_name] = service_item['handler'](service_item.get('settings', {}))
        except Exception as e:
            if service_item['handler'] is None:
                raise Exception("{} service is missing a python library installation.".format(service_name))
            else:
                raise e

    for item in config['files']:
        df = pd.read_csv(item['file_name'])
        handlers[config['handler_name']].insert_df(item['table_name'], df)

