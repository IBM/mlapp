import datetime
import numbers
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pandas as pd
import numpy as np

from mlapp.utils.features.spark import spark_cut, spark_dummies, spark_select_dummies
from mlapp.utils.exceptions.framework_exceptions import UnsupportedFileType


class SparkClassificationFeatureEngineering(object):
    def drop_features(self, data_df, features_to_drop=[]):
        """
        Dropping requested features
        :param data_df: the DataFrame
        :param features_to_drop: list of features names to drop
        :return: data_df after dropping requested featuers
        """
        original_columns = data_df.columns
        filtered_columns_to_drop = filter(lambda x: x in original_columns, features_to_drop)
        return data_df.drop(filtered_columns_to_drop, axis=1)

    def bin_continuous_features(self, data_df, features_to_bin=[]):
        """
        Bin continuous features by the configuration in 'features_to_bin'
        :param data_df: the spark DataFrame
        :param features_to_bin: configuration of bin
        example:

         "features_to_bin":[
          {"name": "feature_name_1", "bins": [5, 15]},
          {"name": "feature_name_2", "bins": [15, 23]}
        ]
        if bins is of length 1 containing one integer n, the feature will be binned to n equal sections.
        if bins is == 'quantile' the feature will be divided according to the quantiles - 0.2, 0.4, 0.6, 0.8
        :return: the DataFrame with requested features transformed
        """
        for feature_to_bin in features_to_bin:
            if feature_to_bin['name'] in data_df.columns:
                if len(feature_to_bin['bins']) == 1:
                    if feature_to_bin['bins'][0] == 'quantile':
                        # bin to quantiles.
                        quantiles = [0.2, 0.4, 0.6, 0.8]
                        min_val = data_df.agg({feature_to_bin['name']: 'min'}).collect()[0][0] -1
                        max_val = data_df.agg({feature_to_bin['name']: 'max'}).collect()[0][0] +1
                        q = data_df.approxQuantile(feature_to_bin['name'], quantiles, relativeError=0.05) #relativeError of 0 is more accurate but more computationaly expensive.
                        bins = [min_val] + q + [max_val]
                        data_df = spark_cut(data_df, feature_to_bin['name'], bins=bins,
                                                 labels=[f'Q_{str(q)[2]}0%' for q in quantiles]+['Q100%'])

                    if isinstance(feature_to_bin['bins'][0], int):
                    # an integer (n) was provided. bin to n identical sections.
                        if len(data_df.select(feature_to_bin['name']).distinct().collect()) > feature_to_bin['bins'][0]:
                            min_val = data_df.agg({feature_to_bin['name']: 'min'}).collect()[0][0] -1
                            max_val = data_df.agg({feature_to_bin['name']: 'max'}).collect()[0][0] +1
                            bins = np.linspace(min_val, max_val, feature_to_bin['bins'][0])
                            data_df = spark_cut(data_df, feature_to_bin['name'], bins=bins,
                                                     labels=range(feature_to_bin['bins'][0]-1))
                else:
                    # bin according to the provided bins.
                    min_val = data_df.agg({feature_to_bin['name']: "min"}).collect()[0][0]
                    max_val = data_df.agg({feature_to_bin['name']: "max"}).collect()[0][0]
                    full_bins = sorted([(min_val - 1)] + feature_to_bin['bins'] + [(max_val + 1)])
                    data_df = spark_cut(
                        data_df, feature_to_bin['name'],
                        bins=full_bins,
                        labels=range(len(full_bins) - 1))
        return data_df

    def handle_y_variable(self, data_df, variable_to_predict, options):
        """
        Transform variable to predict by options given in config
        :param data_df: the DataFrame containing all features and variable to predict
        :param variable_to_predict: the variable to predict columns name
        :param options: options containing the configuration of the transformation for the variable to predict
        example:

        "y_variable": {
          "type": "binary",  # binary/multi/continuous - string
          "categories_labels": ["LABEL_1", "LABEL_2"], # category labels - list
          "continuous_to_category_bins": [-0.5, 0.5, 1.5],  # bins values - list
          "label_to_predict": ["LABEL_1"]  # target label to predict - list
        },

        :return: 'data_df' - without the variable to predict, 'final_y' - the variable to predict after transformation
        """
        # y variable configurations
        y_variable_type = options['type']
        target_label = options['label_to_predict']

        # y variable is binary OR one vs all
        if y_variable_type == 'binary' or (y_variable_type == 'multi' and len(target_label) == 1):
            data_df = spark_dummies(data_df, variable_to_predict)
            data_df = spark_select_dummies(data_df, variable_to_predict, target_label)
            y_column_name = f'{variable_to_predict}_{target_label[0]}'
            data_df = data_df.withColumn(variable_to_predict, F.col(y_column_name)).drop(y_column_name)

        # y variable is multi class
        elif y_variable_type == 'multi' and len(target_label) < len(data_df.select(variable_to_predict).distinct().collect()[0][0]):
            data_df = data_df.withColumn(
                variable_to_predict,
                F.when(F.col(variable_to_predict).isin(target_label), data_df[variable_to_predict]).otherwise('other')
            )

        # y variable continuous
        elif y_variable_type == 'continuous':
            bins = options["continuous_to_category_bins"]
            labels = options["categories_labels"]

            data_df = spark_cut(data_df, variable_to_predict,  bins=bins, labels=labels)

        return data_df

    def transform_dates_to_elapsed_days(self, data, dates_format):
        """
        Go over all features and convert any datetime features to elapsed days till today's date.
        :param data:  spark DataFrame
        :param dates_format: date formats expected in the DataFrame
        :return: the DataFrame with transformed date columns.
        """
        if dates_format is None:
            dates_format = 'yyyyMMdd'

        data_types = data.dtypes
        today = datetime.datetime.now()
        for feature, dtype in data_types:
            if dtype == 'timestamp':
                # Convert timestamp to date.
                data = data.withColumn(feature, F.to_date(feature, dates_format))
                # Convert to elapsed days:
                data.withColumn(feature, F.datediff(F.to_date(F.lit(today)),
                                              F.to_date(feature, "yyyy/MM/dd")))
        return data

    def combine_categorical_features(self, data_df, evaluated_df, sep = '_|_'):
        """
        Combining categories for each feature
        :param data_df: original DataFrame
        :param evaluated_df: calculated evaluated DataFrame for each category for each feature
        :return: DataFrame with combined categories
        """
        features_mapping = {}
        results_df = pd.DataFrame()
        groups = pd.DataFrame.groupby(evaluated_df, 'feature_original_name')
        for feature_original_name, group in groups:
            if group.shape[0] > 1:
                # feature_dummies_df = pd.get_dummies(data_df[feature_original_name])

                filtered_feature_dummies_df = data_df[group['feature']]
                combined_feature = filtered_feature_dummies_df.sum(axis=1)

                # preparing feature output name
                categorical_values = group['feature'].apply(lambda x: x.replace(feature_original_name + "_", ""))
                categorical_values = categorical_values.astype(data_df.columns.dtype)
                feature_output_name = feature_original_name + "_"
                for val in categorical_values:
                    feature_output_name += "_" + str(val)

                # adds combined feature to results DataFrame
                results_df[feature_output_name] = combined_feature
            else:
                # save features mappings
                custom_feature_full_name = group['feature'].iloc[0]
                _, new_feature_value = custom_feature_full_name.split(sep)
                features_mapping[feature_original_name] = [{
                    "name": custom_feature_full_name,
                    "categories": [new_feature_value]
                }]

                results_df[group['feature']] = data_df[group['feature']]
        return results_df, features_mapping

    def fillna_features(self, data, features_handling, default_filling=0, stored_missing_values={}):
        """
        Feature handling with filling missing values strategies
        :param data: DataFrame
        :param features_handling: configuration of how to handle each feature
        :return: updated DataFrame with the requested filling
        """
        methods = {
            "mean": lambda a: F.mean(a),
            "median": lambda a: data.approxQuantile(feature_key,[0.5],0.25)[0],
            # "mode": lambda a: mode(a).mode[0],
            "none": lambda a: float('nan'),
            "nan": lambda a: float('nan')
        }

        if not isinstance(data, DataFrame):
            raise UnsupportedFileType("Error: data type should be spark Dataframe")

        if len(list(stored_missing_values.keys())) > 0:
            data = data.fillna(stored_missing_values)
            missing_values = stored_missing_values
        else:
            missing_values = {}

            specific_features = features_handling.keys()
            for feature_key in data.columns:

                # applying fillna on a feature
                if feature_key in specific_features:
                    filling_missing_value = features_handling[feature_key].get("fillna")
                else:
                    filling_missing_value = default_filling

                if filling_missing_value in methods.keys():
                    filling_missing_value = filling_missing_value.lower()
                    val = data.select(methods[filling_missing_value](data[feature_key])).collect()[0][0]
                    data = data.fillna({feature_key: val})
                    missing_values[feature_key] = val
                elif isinstance(filling_missing_value, numbers.Number):
                    data = data.fillna({feature_key: filling_missing_value})
                    missing_values[feature_key] = filling_missing_value
                else:
                    filling_missing_value = eval('F.' + filling_missing_value)
                    if filling_missing_value is None or np.isnan(filling_missing_value):
                        data = data.fillna({feature_key: float('nan')})
                        missing_values[feature_key] = float('nan')
                    else:
                        val = data.agg(filling_missing_value(feature_key)).collect()[0][0]
                        data = data[feature_key].fillna(val)
                        missing_values[feature_key] = val
        return data, missing_values

    def transform_features(self, data, features_handling):
        '''
        Feature handling with transformation strategies
        :param data:
        :param features_handling:
        :return: DataFrame - updated DataFrame with the requested transformations
        '''

        if not isinstance(data, DataFrame):
            raise p("Error: data type should be a spark Dataframe")

        features = features_handling.keys()
        for feature_key in features:

            # applying transformations
            feature_transformation_methods = features_handling[feature_key].get("transformation", [])
            for feature_transformation_method in feature_transformation_methods:
                data = data.withColumn(feature_key, eval(feature_transformation_method)(data[feature_key]))

            # applying dummies
            feature_dummies_flag = features_handling[feature_key].get("dummies", False)
            if feature_dummies_flag:
                data = spark_dummies(data, feature_key)

        return data

    def get_days_from_date(self, data, date_column, extraction_date):
        datenow = datetime.datetime.strptime(extraction_date, '%Y%m%d')
        transformed_data = date_column
        transformed_data = datenow - pd.to_datetime(data[transformed_data], format='%Y%m%d')
        data[date_column] = transformed_data.dt.days
        return data

    def remove_features_by_null_threshold(self, data, percentage=0.3):
        """
        Removing data with amount of 'nulls' more then the 'percentage'
        :param data: the DataFrame
        :param percentage: percentage - default 30%
        :return: pandas DataFrame
        """
        null_percentages = data.select(
            [(F.count(F.when(F.isnull(c), c)) / data.count()).alias(c) for c in data.columns]).collect()[0]

        n_features = len(data.columns)
        data = data.select([c for c in data.columns if null_percentages[c] < percentage])
        new_n_features = len(data.columns)
        if n_features == new_n_features:
            print("Features number was not changed, did not found null features more than %0.2f percentage" % percentage)
        else:
            print("%d Features has removed, new data shape is (%d,%d)" % ((n_features - new_n_features), data.shape[0], data.shape[1]))
        return data

    def _is_nan(self, x):
        try:
            return np.isnan(x) or x == ""
        except:
            return False

    def _convert_text_to_date_type(self, data_type, field, csv_file, dates_format=None):
        if dates_format is None:
            dates_format = ["%d/%m/%Y", "%Y-%m-%d"]
        if data_type in ['int64', 'float64', 'float32', 'int32']:
            return "NUMERIC", None
        elif data_type == "object":
            for val in csv_file[field]:
                if val is not None and val is not np.nan:
                    for date_format in dates_format:
                        try:
                            datetime.datetime.strptime(val, date_format)
                            return "DATETIME", date_format
                        except:
                            continue
                    return "TEXT", None
                else:
                    return "TEXT", None
        else:
            return "TEXT", None

    def _elapsed_time_from_date(self, x, today, date_format):
        try:
            return np.round(np.abs((today - datetime.datetime.strptime(str(x), date_format)).days) / 365, 1)
        except:
            return np.nan


    def convert_features_to_dummies(self, data, features):
        for feature in features:
            if feature in list(data.columns):
                data = spark_dummies(data, feature)
        return data

    def interact_features(self, data, interact_list):
        for features_pair in interact_list:
            feature_0 = features_pair[0]
            feature_1 = features_pair[1]

            if (feature_0 not in data.columns) or (feature_1 not in data.columns):
                print('Warning: one the features: ' + feature_0 + ',' + feature_1 + 'do not exists in the Data.')
            else:
                name = feature_0 + '_' + feature_1
                data = data.withColumn(name, (F.col(feature_0)) * (F.col(feature_0)))
        return data

