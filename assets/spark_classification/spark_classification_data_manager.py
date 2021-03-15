import numpy as np
import os
from mlapp.handlers.instance import spark_handler
from assets.spark_classification.spark_classification_feature_engineering import SparkClassificationFeatureEngineering
from mlapp.managers import DataManager, pipeline
from mlapp.utils.exceptions.base_exceptions import ConfigKeyError


class SparkClassificationDataManager(DataManager):
    # -------------------------------- custom init function ---------------------------------------
    def __init__(self, config, *args, **kwargs):
        DataManager.__init__(self, config, *args, **kwargs)

        # Custom initiates
        if self.data_settings.get('data_handling', None) is None:
            raise ConfigKeyError("'data_handling' is needed in model configuration file!!!")

        # Feature engineering helper class
        self.feature_engineering_instance = SparkClassificationFeatureEngineering()

    # -------------------------------------- train methods -------------------------------------------
    @pipeline
    def load_train_data(self,*args):
        return self._load_data()

    @pipeline
    def clean_train_data(self, data):
        print('------------- CLEAN TRAIN DATA -------------')
        data, missing_values = self._clean_data(data)
        self.save_metadata('missing_values', missing_values)
        return data

    @pipeline
    def transform_train_data(self, data):
        results_df = self._transform_data(data)
        print("------------------------- return final features to train -------------------------")
        return results_df

    # ------------------------------------- forecast methods -----------------------------------------
    @pipeline
    def load_forecast_data(self, *args):
        return self._load_data()

    @pipeline
    def clean_forecast_data(self, data):
        print('------------- CLEAN FORECAST DATA -------------')
        data, missing_values = self._clean_data(data)
        return data

    @pipeline
    def transform_forecast_data(self, data):
        data_handling = self.data_settings.get('data_handling', {})
        dates_transformation = data_handling.get("dates_transformation", {})
        return self._transform_data(data, dates_transformation=dates_transformation)

    def transform_data_for_exploration(self, data):
        data_handling = self.data_settings.get('data_handling', {})

        for col in data.columns:
            if data[col].dtype == np.bool:
                data[col] = data[col].astype(int)

        data_df = self.feature_engineering_instance.transform_features(data, data_handling.get('features_handling'))

        print("------------------------- handling y variable -------------------------")
        final_y = data_df[self.data_settings["variable_to_predict"]]
        data_df = data_df.drop(self.data_settings["variable_to_predict"], axis=1)

        print("------------------------- merge final features with y value -------------------------")
        results_df = data_df.join(final_y.rename(self.data_settings["variable_to_predict"]))

        print("------------------------- return final features to train -------------------------")
        return results_df

    # ------------------------------ helper methods for train/forecast --------------------------------
    def _load_data(self):
        local_path = os.path.join(os.getcwd(), self.data_settings["local_data_csvs"][0].get("path", ""))
        return spark_handler('LOCAL-SPARK').load_csv_file(local_path, inferSchema=True)

    def _clean_data(self, data):
        data_handling = self.data_settings.get('data_handling')
        set_features_index = data_handling.get('set_features_index', [])
        if set_features_index:
            data = data.set_index(set_features_index)

        features_for_train = data_handling.get('features_for_train', [])
        features_to_remove = data_handling.get('features_to_remove', [])

        if features_to_remove:
            for feature_to_remove in features_to_remove:
                if feature_to_remove in data.columns and (
                        feature_to_remove != self.data_settings["variable_to_predict"]):
                    data = data.drop(feature_to_remove)

        if features_for_train:
            features_for_train = list(filter(lambda x: x not in features_to_remove, features_for_train))
            features_for_train += [self.data_settings["variable_to_predict"]]
            data = data[features_for_train]

        print("------------------------- Removing high percentage of null features -------------------------")
        data = self.feature_engineering_instance.remove_features_by_null_threshold(data, data_handling.get(
            'feature_remove_by_null_percentage', 0.3))

        print("------------------------- fill na -------------------------")
        stored_missing_values = self.get_metadata('missing_values', default_value={})

        data, missing_values = self.feature_engineering_instance.fillna_features(
            data, data_handling.get('features_handling', {}), data_handling.get('default_missing_value', 0),
            stored_missing_values)

        return data, missing_values

    def _transform_data(self, data_df, dates_transformation=None):
        data_handling = self.data_settings.get('data_handling', {})
        # Convert boolean columns to int:
        data_df = data_df.select(
            [data_df[colName].cast('int') if colType == 'boolean' else data_df[colName] for colName, colType in data_df.dtypes])

        data_df = self.feature_engineering_instance.transform_features(data_df, data_handling.get('features_handling'))
        dates_transformation = data_handling.get('dates_transformation', {})
        if dates_transformation:
            for col in dates_transformation.get('columns', []):
                pass
                # TODO:
                # data_df = self.feature_engineering_instance.get_days_from_date(
                #     data_df, col,
                #     dates_transformation.get('extraction_date') if extraction_date is None else extraction_date)

        print("------------------------- handling y variable -------------------------")
        data_df = self.feature_engineering_instance.handle_y_variable(
            data_df, self.data_settings["variable_to_predict"], data_handling['y_variable'])

        print("------------------------- feature interactions -------------------------")
        interactions_list = data_handling.get('features_interactions', [])
        data_df = self.feature_engineering_instance.interact_features(data_df, interactions_list)

        print("---------------------- Convert dates to days elapsed from date -------------")
        data_df = self.feature_engineering_instance.transform_dates_to_elapsed_days(
            data_df, data_handling.get("dates_format", 'yyyyMMdd'))

        print("------------------------- bin continuous features -------------------------")
        data_df = self.feature_engineering_instance.bin_continuous_features(
            data_df, data_handling.get("features_to_bin", []))

        print("---------------------- transform categorical to dummy variables -------------")
        data_df = self.feature_engineering_instance.convert_features_to_dummies(
            data_df, data_handling.get("features_to_convert_to_dummies", []))

        data_df.cache()
        return data_df

    def load_target_data(self, *args):
        raise NotImplementedError()

