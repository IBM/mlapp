from mlapp.handlers.instance import database_handler, file_storage_handler
from mlapp.managers import DataManager, pipeline
import pandas as pd
import os
import numpy as np


class AdvancedRegressionDataManager(DataManager):
    # -------------------------------------- train methods -------------------------------------------
    @pipeline
    def load_train_data(self, *args):
        return self._load_data_inner(self.data_settings.get('data_sources', []))

    @pipeline
    def clean_train_data(self, data):
        missing_values = {}
        data_handling = self.data_settings.get('data_handling', {})
        features_handling = self.data_settings.get('features_handling', {})
        null_percentage = data_handling.get("feature_remove_by_null_percentage", 0.5)
        data = data.loc[:, data.isnull().mean() < null_percentage]

        for feature in features_handling.keys():
            missing_values[feature] = eval(features_handling[feature].get("fillna", "np.mean"))(data[feature].values)

        default_missing_features = list(set(data.columns).difference(set(list(features_handling.keys()))))
        default_missing_values = data[default_missing_features].mean(axis=0).to_dict()
        missing_values.update(default_missing_values)
        self.save_metadata('missing_values', missing_values)

        data = data.fillna(missing_values)
        return data

    @pipeline
    def transform_train_data(self, data):
        return self._transform_data(data)

    # ------------------------------------- forecast methods -----------------------------------------
    @pipeline
    def load_forecast_data(self, *args):
        # read data for forecasting
        data = self._load_data_inner(self.data_settings.get('data_sources', []))

        # creating prediction data for example
        if self.data_settings.get('generate_forecast_data') == 'index_based':
            idx = self.data_settings.get('generate_forecast_data_indices', data.index)
            data = data.loc[idx]
            data.drop([self.model_settings['variable_to_predict']], axis=1, inplace=True)
        else:
            data.drop([self.model_settings['variable_to_predict']], axis=1, inplace=True)
            data = pd.DataFrame(
                data=np.random.normal(data.mean().mean(), data.std().mean(), data.shape)[0:10], columns=data.columns)
        return data
    
    @pipeline
    def clean_forecast_data(self, data):
        return data

    @pipeline
    def transform_forecast_data(self, data):
        return self._transform_data(data)

    # ------------------------- load helper function for train/forecast ------------------------------
    def _load_data_inner(self, sources):
        data = pd.DataFrame()
        for source_type in sources.keys():
            source_data = pd.DataFrame()

            # local file
            if source_type == 'local':
                for file_path in sources[source_type]['file_paths']:
                    local_path = os.path.join(os.getcwd(), file_path)
                    source_data = pd.concat([source_data, self._load_data_from_file(local_path)])

            # database
            if source_type == 'db':
                source_data = self._load_data_from_db(sources[source_type]['query'], sources[source_type]['args'])

            # s3 file store
            if source_type == 's3':
                for bucket in sources[source_type]['buckets']:
                    for file_key in sources[source_type]['buckets'][bucket]['file_keys']:
                        source_local_file = self._load_file_from_file_store(bucket, file_key)
                        source_data = pd.concat([source_data, pd.read_csv(source_local_file)])

                        # removing local file
                        os.remove(source_local_file)

            data = pd.concat([data, source_data])

        return data

    @staticmethod
    def _load_data_from_file(path):
        return pd.read_csv(path, encoding='ISO-8859-1')

    @staticmethod
    def _load_data_from_db(query, params=None):
        if params is None:
            params = []

        return database_handler('POSTGRES').get_df(query, params)

    def _load_file_from_file_store(self, bucket, file_key):
        unicode_file_name = str(file_key)
        to_path = os.path.join(self.local_storage_path, unicode_file_name)
        file_storage_handler('MINIO').download_file(bucket, unicode_file_name, to_path)
        return to_path

    def _transform_data(self,data):
        data_handling = self.data_settings.get('data_handling', {})

        # interactions
        if data_handling.get('interactions', False):
            columns_list = [c for c in list(data.columns) if c != 'target']
            for col1 in columns_list:
                for col2 in columns_list:
                    if col1 != col2:
                        name = str(col1) + '_' + str(col2)
                        reverse_name = str(col2) + '_' + str(col1)
                        if reverse_name not in list(data.columns):
                            data[name] = (data[col1] + 1) * (data[col2] + 1)

        # binning
        for feature_to_bin in data_handling.get("features_to_bin", []):
            full_bins = [data[feature_to_bin['name']].min() - 1] + \
                        feature_to_bin['bins'] + [data[feature_to_bin['name']].max() + 1]

            data[feature_to_bin['name'] + '_binned'] = pd.cut(
                data[feature_to_bin['name']],
                bins=full_bins,
                labels=range(len(full_bins) - 1)).astype(float)

        # transformation
        for col in data_handling.get("features_handling", {}).keys():
            transformation_array = data_handling["features_handling"][col].get("transformation", [])
            # applying transformations
            for feature_transformation_method in transformation_array:
                data[col + '_' + feature_transformation_method] = eval(feature_transformation_method)(
                    data[col])

        # dropping features
        features_to_remove = data_handling.get('features_to_remove', [])
        if len(features_to_remove) > 0:
            data = data.drop([feature for feature in features_to_remove if feature in data.columns], axis=1)
        return data

    @pipeline
    def load_actuals_data(self, *args):
        index_to_y_true = self.model_settings.get('index_to_y_true', [])
        df = pd.DataFrame(index_to_y_true, index=[0])
        return df

    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()