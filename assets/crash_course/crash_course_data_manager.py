from mlapp.managers import DataManager, pipeline
import pandas as pd
from mlapp.utils.features.pandas import extend_dataframe


class CrashCourseDataManager(DataManager):
    @pipeline
    def load_train_data(self, *args):
        data = pd.read_csv(self.data_settings["file_path"])
        return data

    @pipeline
    def clean_train_data(self, data):
        # Drop samples with missing label
        data.dropna(inplace=True, subset=[self.model_settings['target']])

        # Extract the list of features excluding the variable to predict
        features = list(data.columns)
        features.remove(self.model_settings['target'])

        # Calculate the mean value for each feature
        default_values_for_missing = data[features].mean(axis=0).to_dict()

        # Fill any missing values using the previous calculation
        data = data.fillna(default_values_for_missing)

        # Store the calculated missing values
        self.save_metadata('missing_values', default_values_for_missing)
        return data

    @pipeline
    def transform_train_data(self, data):
        # apply transformations
        data = self._transform_data(data)

        # save data frame
        self.save_dataframe('features', data)

        return data

    @pipeline
    def load_forecast_data(self,*args):
        data = pd.read_csv(self.data_settings["file_path"])
        return data

    @pipeline
    def clean_forecast_data(self, data):
        # get the missing values
        default_values_for_missing = self.get_metadata('missing_values', default_value={})

        # fill the missing values
        data = data.fillna(default_values_for_missing)
        return data

    @pipeline
    def transform_forecast_data(self, data):
        # apply transformations
        data = self._transform_data(data)
        return data

    def _transform_data(self, data):
        # extend data frame
        conf_extend_dataframe = self.data_settings.get('conf_extend_dataframe', {})
        data = extend_dataframe(
            data,
            y_name_col=self.model_settings.get('target'),
            lead_order=conf_extend_dataframe.get('lead', 0),
            lag_order=conf_extend_dataframe.get('lag', 0),
            power_order=conf_extend_dataframe.get('power', 0),
            log=conf_extend_dataframe.get('log', False),
            exp=conf_extend_dataframe.get('exp', False),
            sqrt=conf_extend_dataframe.get('sqrt', False),
            poly_degree=conf_extend_dataframe.get('interactions', 0),
            inverse=conf_extend_dataframe.get('inverse', False)
        )
        return data

    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()
