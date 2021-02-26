import numbers
from scipy.stats import mode
import pandas as pd
import numpy as np
import datetime

from mlapp.utils.exceptions.framework_exceptions import UnsupportedFileType


class ClassificationFeatureEngineering(object):
    def drop_features(self, data_df, features_to_drop=None):
        """
        Dropping requested features
        :param data_df: the DataFrame
        :param features_to_drop: list of features names to drop
        :return: data_df after dropping requested featuers
        """
        if not features_to_drop:
            features_to_drop = []

        original_columns = data_df.columns
        filtered_columns_to_drop = filter(lambda x: x in original_columns, features_to_drop)
        return data_df.drop(filtered_columns_to_drop, axis=1)

    def bin_continuous_features(self, data_df, features_to_bin=None):
        """
        Bin continuous features by the configuration in 'features_to_bin'
        :param data_df: the DataFrame
        :param features_to_bin: configuration of bin
        example:

         "features_to_bin":[
          {"name": "feature_name_1", "bins": [5, 15]},
          {"name": "feature_name_2", "bins": [15, 23]}
        ]

        :return: the DataFrame with requested features transformed
        """
        if not features_to_bin:
            features_to_bin = []

        for feature_to_bin in features_to_bin:
            if feature_to_bin['name'] in data_df.columns:
                full_bins = [data_df[feature_to_bin['name']].min() - 1] + feature_to_bin['bins'] + [data_df[feature_to_bin['name']].max() + 1]
                data_df[feature_to_bin['name']] = pd.cut(
                    data_df[feature_to_bin['name']],
                    bins=full_bins,
                    labels=range(len(full_bins) - 1)).astype(float)
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
        y_df = data_df[variable_to_predict]
        final_y = pd.DataFrame()
        y_variable_type = options['type']
        target_label = options['label_to_predict']

        # y variable is binary OR one vs all
        if y_variable_type == 'binary' or (y_variable_type == 'multi' and len(target_label) == 1):
            y_dummies = pd.get_dummies(y_df)
            final_y = y_dummies[target_label[0]]

        # y variable is multi class
        elif y_variable_type == 'multi' and len(target_label) < len(y_df.unique()):
            final_y = y_df.apply(lambda x: x if x in target_label else "other")

            # Example for indexing the labels
            # labels_y = final_y.copy()
            # for i in range(len(target_model)):
            #     labels_y = labels_y.apply(lambda x: i + 1 if x == target_model[i] else x)
            # final_y = labels_y.apply(lambda x: 0 if not type(x)==int else x)

        elif y_variable_type == 'continuous':
            bins = options["continuous_to_category_bins"]
            labels = options["categories_labels"]
            final_y = pd.cut(y_df, bins=bins, labels=labels)
        else:
            final_y = y_df

        data_df = data_df.drop(variable_to_predict, axis=1)
        return data_df, final_y

    def transform_and_split_features_to_categorical_and_continuous(self, data, dates_format=None, auto_bin_continuous_features=False, max_categories_num=10):
        """
        Transforming DataFrame features by their value types
        :param data: the DataFrame
        :param dates_format: date formats expected in the DataFrame
        :param auto_bin_continuous_features: whether to bin continuous features automatically
        :param max_categories_num: max unique values in a feature before deciding to auto bin
        :return: the DataFrame with transformed date columns, lists of features by their type, and binned features
        """
        if dates_format is None:
            dates_format = ["%d/%m/%Y", "%Y-%m-%d"]

        data_types = data.dtypes
        today = datetime.datetime.now()
        continuous_columns = []
        continuous_bins = {}
        categorical_columns = []
        binary_columns = []
        for feature, curr_type in data_types.iteritems():
            mysql_type, date_format = self._convert_text_to_date_type(curr_type, feature, data, dates_format)
            if mysql_type == "DATETIME": # converting features from datetime to time_passed_from_date
                data[feature] = data[feature].apply(
                    lambda x: x if self._is_nan(x) else self._elapsed_time_from_date(x, today, date_format))
                if auto_bin_continuous_features:
                    continuous_bins[feature] = np.sort(list(
                        {
                            min(data[feature]) - 1,
                            np.quantile(data[feature].dropna(), 0.2),
                            np.quantile(data[feature].dropna(), 0.4),
                            np.quantile(data[feature].dropna(), 0.6),
                            np.quantile(data[feature].dropna(), 0.8),
                            max(data[feature]) + 1
                        }))
                else:
                    continuous_columns += [feature]
            elif mysql_type == 'NUMERIC':
                unique_values = data[feature].dropna().unique()
                if len(unique_values) == 1:
                    data = data.drop(feature, axis=1)
                elif len(unique_values) == 2:
                    binary_columns += [feature]
                elif (2 < len(unique_values) <= max_categories_num) and auto_bin_continuous_features:
                    categorical_columns += [feature]
                elif auto_bin_continuous_features:
                    continuous_bins[feature] = np.sort(list(
                        {
                            min(data[feature]) - 1,
                            np.quantile(data[feature].dropna(), 0.2),
                            np.quantile(data[feature].dropna(), 0.4),
                            np.quantile(data[feature].dropna(), 0.6),
                            np.quantile(data[feature].dropna(), 0.8),
                            max(data[feature]) + 1
                        }))
                else:
                    continuous_columns += [feature]
            else: # mysql_type == TEXT
                categorical_columns += [feature]

        return data, categorical_columns, continuous_columns, binary_columns, continuous_bins

    def combine_categorical_features(self, data_df, evaluated_df, sep='_|_'):
        """
        Combining categories for each feature
        :param data_df: original DataFrame
        :param evaluated_df: calculated evaluated DataFrame for each category for each feature
        :param sep: separation string
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

    def fillna_features(self, data, features_handling, default_filling=0, missing_values=None):
        """
        Feature handling with filling missing values strategies
        :param data: DataFrame
        :param features_handling: configuration of how to handle each feature
        :return: updated DataFrame with the requested filling
        """
        if missing_values:
            missing_values = {}

        methods = {
            "mean": lambda a: np.mean(a),
            "median": lambda a: np.median(a),
            "mode": lambda a: mode(a).mode[0],
            "none": lambda a: np.nan,
            "nan": lambda a: np.nan
        }

        if not isinstance(data, pd.DataFrame):
            raise UnsupportedFileType("data type should be Dataframe")

        if len(list(missing_values.keys())) > 0:
            data.fillna(missing_values, inplace=True)
        else:
            missing_values = {}
            specific_features = features_handling.keys()
            for feature_key in data.columns:

                # applying fill na on a feature
                if feature_key in specific_features:
                    filling_missing_value = features_handling[feature_key].get("fillna")
                else:
                    filling_missing_value = default_filling

                if filling_missing_value in methods.keys():
                    filling_missing_value = filling_missing_value.lower()
                    val = methods[filling_missing_value](data[feature_key])
                    data[feature_key].fillna(val, inplace=True)
                    missing_values[feature_key] = val
                elif isinstance(filling_missing_value, numbers.Number):
                    data[feature_key].fillna(filling_missing_value, inplace=True)
                    missing_values[feature_key] = filling_missing_value
                else:
                    filling_missing_value = eval(filling_missing_value)
                    if filling_missing_value is None or filling_missing_value == np.nan:
                        data[feature_key] = data[feature_key].fillna(methods["none"], inplace=True)
                        missing_values[feature_key] = np.nan
                    else:
                        val = filling_missing_value(data[feature_key])
                        data[feature_key].fillna(val, inplace=True)
                        missing_values[feature_key] = val
        return data, missing_values


    def transform_features(self, data,features_handling):
        '''
        Feature handling with transformation strategies
        :param data:
        :param features_handling:
        :return: DataFrame - updated DataFrame with the requested transformations
        '''

        if not isinstance(data, pd.DataFrame):
            raise UnsupportedFileType("data type should be Dataframe")

        features = features_handling.keys()
        for feature_key in features:

            # applying transformations
            feature_transformation_methods = features_handling[feature_key].get("transformation", [])
            for feature_transformation_method in feature_transformation_methods:
                data[feature_key] = eval(feature_transformation_method)(data[feature_key])

            # applying dummies
            feature_dummies_flag = features_handling[feature_key].get("dummies", False)
            if feature_dummies_flag:
                dummies_df = pd.get_dummies(data[feature_key], dummy_na=False)
                dummies_df['index'] = data.index.values
                dummies_df = dummies_df.set_index('index')
                data = pd.concat([data, dummies_df], axis=1)
                data = data.drop(feature_key, axis=1)

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
        # data = data.loc[:, data.isnull().mean() < percentage]
        n_features = data.shape[1]
        data = data.loc[:, data.isnull().mean() < percentage]
        new_n_features = data.shape[1]
        if n_features == new_n_features:
            print("Features number did not changed, did not found null features more than %0.2f percentage" % percentage)
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
