from mlapp.handlers.instance import spark_handler
from mlapp.managers import DataManager, pipeline
import pyspark.sql.functions as F
from pyspark.ml.feature import Bucketizer


class SparkRegressionDataManager(DataManager):
    @pipeline
    def load_train_data(self, *args):
        return self._load_data()

    @pipeline
    def load_forecast_data(self,*args):
        return self._load_data()

    @pipeline
    def clean_train_data(self, data):
        return self._clean_data(data)

    @pipeline
    def clean_forecast_data(self, data):
        missing_values = self.get_metadata('missing_values', {})
        return self._clean_data(data, missing_values)

    @pipeline
    def transform_train_data(self, data):
        return self._transform_data(data)

    @pipeline
    def transform_forecast_data(self, data):
        return self._transform_data(data)

    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()

    # ------------------------- private functions for load/clean/transform ------------------------------

    def _load_data(self):
        return spark_handler('LOCAL-SPARK').load_csv_file(self.data_settings["local_file_path"], inferSchema=True)

    def _clean_data(self, data, stored_missing_values=None):
        missing_values = {}

        data_handling = self.data_settings.get('data_handling', {})
        features_handling = data_handling.get('features_handling', {})

        # remove features by null percentage
        null_percentage = data_handling.get("feature_remove_by_null_percentage", 0.5)
        null_percentages = data.select(
            [(F.count(F.when(F.isnull(c), c)) / data.count()).alias(c) for c in data.columns]).collect()[0]
        data = data.select([c for c in data.columns if null_percentages[c] < null_percentage])

        # filling missing values by function/value
        if len(features_handling.keys()) > 0:
            missing_values = {
                k: v['fillna'] if not isinstance(v.get('fillna', 'mean'), str) else
                data.agg((eval('F.' + v.get('fillna', 'mean')))(k)).collect()[0][0]
                for (k, v) in features_handling.items()
            }

        # filling default missing features by mean
        default_missing_features = list(set(data.columns).difference(set(list(features_handling.keys()))))
        default_missing_values = data.select([F.mean(c).alias(c) for c in default_missing_features]).collect()[0]
        missing_values.update({c: default_missing_values[c] for c in default_missing_features})
        self.save_metadata('missing_values', missing_values)

        if stored_missing_values is not None:
            data = data.fillna(stored_missing_values)
        else:
            data = data.fillna(missing_values)
        return data

    def _transform_data(self, data):
        data_handling = self.data_settings.get('data_handling', {})

        # interactions
        if data_handling.get('interactions', False):
            columns_list = list(data.columns)
            columns_list.remove(self.model_settings['variable_to_predict'])
            for col1 in columns_list:
                for col2 in columns_list:
                    if col1 != col2:
                        name = str(col1) + '_' + str(col2)
                        reverse_name = str(col2) + '_' + str(col1)
                        if reverse_name not in list(data.columns):
                            data = data.withColumn(name, (F.col(col1) + 1) * (F.col(col2) + 1))

        # binning
        for feature_to_bin in data_handling.get("features_to_bin", []):
            min_val = data.agg({feature_to_bin['name']: "min"}).collect()[0][0]
            max_val = data.agg({feature_to_bin['name']: "max"}).collect()[0][0]
            full_bins = [(min_val - 1)] + feature_to_bin['bins'] + [(max_val + 1)]

            bucketizer = Bucketizer(splits=full_bins,
                                    inputCol=feature_to_bin['name'],
                                    outputCol=feature_to_bin['name'] + '_binned')

            data = bucketizer.transform(data)

        # transformation
        for col in data_handling.get("features_handling", {}).keys():
            transformation_array = data_handling["features_handling"][col].get("transformation", [])
            # applying transformations
            for feature_transformation_method in transformation_array:
                data = data.withColumn(
                    col + '_' + feature_transformation_method, eval('F.' + feature_transformation_method)(col))

        # dropping features
        features_to_remove = data_handling.get('features_to_remove', [])
        if len(features_to_remove) > 0:
            data = data.drop(*[feature for feature in features_to_remove if feature in data.columns])
        return data


