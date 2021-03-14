from mlapp.managers import DataManager, pipeline
import pandas as pd
import os
import numpy as np


class BasicRegressionDataManager(DataManager):
    # -------------------------------------- train methods -------------------------------------------
    @pipeline
    def load_train_data(self, *args):
        local_path = os.path.join(os.getcwd(), self.data_settings.get('local_file_path'))
        data = self._load_data_from_file(local_path)
        return data

    @pipeline
    def clean_train_data(self, data):
        data_handling = self.data_settings.get('data_handling', {})
        null_percentage = data_handling.get("feature_remove_by_null_percentage", 0.5)
        data = data.loc[:, data.isnull().mean() < null_percentage]
        missing_values = data.mean(axis=0).to_dict()
        data = data.fillna(missing_values)
        self.save_metadata('missing_values', missing_values)
        return data

    @pipeline
    def transform_train_data(self, data):
        return self._transform_data(data)

    # ------------------------------------- forecast methods -----------------------------------------
    @pipeline
    def load_forecast_data(self, *args):
        local_path = os.path.join(os.getcwd(), self.data_settings.get('local_file_path'))
        data = self._load_data_from_file(local_path)
        
        # creating predict data for example
        data.drop([self.model_settings['variable_to_predict']], axis=1, inplace=True)
        data = pd.DataFrame(data=np.random.normal(data.mean().mean(), data.std().mean(), data.shape)[0:10],
                            columns=data.columns)
        return data

    @pipeline
    def clean_forecast_data(self, data):
        missing_values = self.get_metadata('missing_values', {})
        data = data.fillna(missing_values)
        return data

    @pipeline
    def transform_forecast_data(self, data):
        return self._transform_data(data)

    # ------------------------- load helper function for train/forecast ------------------------------
    @staticmethod
    def _load_data_from_file(path):
        return pd.read_csv(path, encoding='ISO-8859-1')

    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()

    def _transform_data(self, data):
        data_handling = self.data_settings.get('data_handling', {})

        # dropping features
        features_to_remove = data_handling.get('features_to_remove', [])
        if len(features_to_remove) > 0:
            data = data.drop([feature for feature in features_to_remove if feature in data.columns], axis=1)
        return data




