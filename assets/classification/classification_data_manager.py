import numpy as np
import os
import pandas as pd
from mlapp.utils.features.pandas import evaluate_df_with_binary_output, interact_features
from assets.classification.classification_feature_engineering import ClassificationFeatureEngineering
from mlapp.managers import DataManager, pipeline


class ClassificationDataManager(DataManager):
    # -------------------------------- custom init function ---------------------------------------
    def __init__(self, config, *args, **kwargs):
        DataManager.__init__(self, config, *args, **kwargs)

        # Custom initiates
        if self.data_settings.get('data_handling', None) is None:
            raise Exception("'data_handling' is needed in model configuration file!!!")

        # Feature engineering helper class
        self.feature_engineering_instance = ClassificationFeatureEngineering()

        # actions for continuous features
        self.AUTO_BIN = "auto_bin"
        self.KEEP_AS_IS = "keep_as_is"
        self.REMOVE = "remove"

    # -------------------------------------- train methods -------------------------------------------
    @pipeline
    def load_train_data(self, *args):
        local_path = os.path.join(os.getcwd(), self.data_settings["local_data_csvs"][0].get("path", ""))
        data = self._load_data(local_path)
        return data

    @pipeline
    def clean_train_data(self, data):
        print('------------- CLEAN TRAIN DATA -------------')
        data, missing_values = self._clean_data(data)
        self.save_metadata('missing_values', missing_values)
        return data

    @pipeline
    def transform_train_data(self, data):
        results_df = self._transform_data(data)
        # ------------------------- return final features to train -------------------------
        print("------------------------- return final features to train -------------------------")
        return results_df

    # ------------------------------------- forecast methods -----------------------------------------
    @pipeline
    def load_forecast_data(self,*args):
        print('------------- LOAD FORECAST DATA -------------')
        local_path = os.path.join(os.getcwd(),self.data_settings["local_data_csvs"][0].get("path", ""))
        data = self._load_data(local_path)
        print('------------- DONE LOAD FORECAST DATA -------------')
        return data

    @pipeline
    def clean_forecast_data(self, data):
        print('------------- CLEAN FORECAST DATA -------------')
        data, missing_values = self._clean_data(data)
        return data

    @pipeline
    def transform_forecast_data(self, data):
        data_handling = self.data_settings.get('data_handling', {})
        dates_transformation = data_handling.get("dates_transformation", {})
        categorical_features = self.get_metadata('evaluator_features_mappings')
        return self._transform_data(
            data,
            mandatory_categorical_features=categorical_features,
            extraction_date=dates_transformation.get('extraction_date'))

    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()

    def transform_data_for_exploration(self, data):
        data_handling = self.data_settings.get('data_handling', {})

        for col in data.columns:
            if data[col].dtype == np.bool:
                data[col] = data[col].astype(int)

        data_df = self.feature_engineering_instance.transform_features(data, data_handling.get('features_handling'))

        print("------------------------- handling y variable -------------------------")
        final_y = data_df[self.model_settings.get("variable_to_predict", self.data_settings.get("variable_to_predict"))]
        data_df = data_df.drop(self.model_settings.get("variable_to_predict", self.data_settings.get("variable_to_predict")), axis=1)

        print("------------------------- feature interactions -------------------------")
        interactions_list = data_handling.get('features_interactions', [])
        data_df = interact_features(data_df, interactions_list)

        print("------------------------- merge final features with y value -------------------------")
        results_df = data_df.join(final_y.rename(self.model_settings.get("variable_to_predict", self.data_settings.get("variable_to_predict"))))

        # ------------------------- return final features to train -------------------------
        print("------------------------- return final features to train -------------------------")
        return results_df

    # ------------------------------ helper methods for train/forecast --------------------------------
    @staticmethod
    def _load_data(path):
        data = pd.read_csv(path, encoding='ISO-8859-1')
        return data

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
                        feature_to_remove != self.model_settings.get("variable_to_predict", self.data_settings.get("variable_to_predict"))):
                    data = data.drop(feature_to_remove, axis=1)

        if features_for_train:
            features_for_train = list(filter(lambda x: x not in features_to_remove, features_for_train))
            features_for_train += [self.model_settings.get("variable_to_predict", self.data_settings.get("variable_to_predict"))]
            data = data[features_for_train]

        # ------------------------- Removing high percentage of null features -------------------------
        print("------------------------- Removing high percentage of null features -------------------------")
        data = self.feature_engineering_instance.remove_features_by_null_threshold(data, data_handling.get(
            'feature_remove_by_null_percentage', 0.3))

        print("------------------------- fill na -------------------------")
        missing_values = self.get_metadata('missing_values', {})
        data, missing_values = self.feature_engineering_instance.fillna_features(
            data, data_handling.get('features_handling', {}), data_handling.get('default_missing_value', 0),
            missing_values)

        return data, missing_values

    def _transform_data(self, data_df, mandatory_categorical_features=None, extraction_date=None):
        data_handling = self.data_settings.get('data_handling', {})
        evaluator_settings = data_handling.get('evaluator_settings', {})

        for col in data_df.columns:
            if data_df[col].dtype == np.bool:
                data_df[col] = data_df[col].astype(int)

        action_for_continuous_features = data_handling.get('action_for_continuous_features', self.AUTO_BIN)

        data_df = self.feature_engineering_instance.transform_features(data_df, data_handling.get('features_handling'))
        dates_transformation = data_handling.get('dates_transformation', {})
        if dates_transformation:
            for col in dates_transformation.get('columns', []):
                data_df = self.feature_engineering_instance.get_days_from_date(
                    data_df, col,
                    dates_transformation.get('extraction_date') if extraction_date is None else extraction_date)

        # ------------------------- handling y variable -------------------------
        print("------------------------- handling y variable -------------------------")
        data_df, final_y = self.feature_engineering_instance.handle_y_variable(
            data_df, self.model_settings.get("variable_to_predict", self.data_settings.get("variable_to_predict")),
            data_handling['y_variable'])

        print("------------------------- feature interactions -------------------------")
        interactions_list = data_handling.get('features_interactions', [])
        data_df = interact_features(data_df, interactions_list)

        # ------------------------- bin some features -------------------------
        print("------------------------- bin continuous features -------------------------")
        data_df = self.feature_engineering_instance.bin_continuous_features(
            data_df, data_handling.get("features_to_bin", []))

        # -------------- Transform and split features to categorical and continues features ----------------------------
        print("---------------------- transform and split features to categorical and continues features -------------")
        data_df, categorical_columns, continuous_columns, binary_columns, continuous_bins = \
            self.feature_engineering_instance.transform_and_split_features_to_categorical_and_continuous(
                data_df,
                data_handling.get("dates_format", ["%d/%m/%Y", "%Y-%m-%d"]),
                action_for_continuous_features == self.AUTO_BIN)

        # TODO: take out the evaluator to a different file
        # ------------------ Call evaluator with continuous features bins and categorical values -----------------------
        print("--------------- Call evaluator with continues features bins and categorical values --------------------")
        store_evaluator_features = evaluator_settings.get("store_evaluator_features", False)

        # non_continuous_columns = binary_columns + categorical_columns + list(continuous_bins.keys())
        non_continuous_columns = categorical_columns + list(continuous_bins.keys())
        evaluator_features_sep = evaluator_settings.get("evaluator_features_separator", '_|_')
        dummies_data_df = pd.get_dummies(data_df, columns=non_continuous_columns, prefix_sep=evaluator_features_sep)

        if mandatory_categorical_features is None and data_handling.get('use_evaluator', True):
            evaluated_df = evaluate_df_with_binary_output(
                data_df[non_continuous_columns], final_y, categorical_features=categorical_columns,
                bins_dict=continuous_bins)

            # store evaluator
            self.save_dataframe('evaluator_features', evaluated_df)

            # -- Evaluator: Filter returned features from evaluator according to filter_evaluator_threshold ------------
            print("---- Evaluator: Filter returned features from evaluator according to filter_evaluator_threshold ---")

            if len(final_y.unique()) == 2:
                y_precision = np.true_divide(final_y.sum(), len(final_y))
            else:
                y_precision = 0

            filter_evaluator_threshold = evaluator_settings.get('filter_evaluator_threshold', .25)
            min_threshold = filter_evaluator_threshold + y_precision
            max_threshold = 1
            diff_threshold = max_threshold - min_threshold
            mid_threshold = (diff_threshold / 2) + min_threshold
            lower_features = evaluated_df[
                (evaluated_df['precision_by_1'] > min_threshold) & (evaluated_df['precision_by_1'] < mid_threshold)]
            higher_features = evaluated_df[
                (evaluated_df['precision_by_1'] >= mid_threshold) & (evaluated_df['precision_by_1'] < max_threshold)]

            # ------------------------- Combine filtered evaluator features -------------------------
            print("------------------------- Combine filtered evaluator features -------------------------")
            combined_lower_features_df, lower_features_mapping = self.feature_engineering_instance.\
                combine_categorical_features(dummies_data_df,lower_features,sep=evaluator_features_sep)

            combined_higher_features_df, higher_features_mapping = self.feature_engineering_instance.\
                combine_categorical_features(dummies_data_df,higher_features,sep=evaluator_features_sep)

            combined_df = pd.concat([combined_lower_features_df, combined_higher_features_df], axis=1)

            # creates evaluator_features_mappings
            evaluator_features_mappings = {**lower_features_mapping,**higher_features_mapping}
            for feature in set(lower_features_mapping.keys()) & set(higher_features_mapping.keys()):
                evaluator_features_mappings[feature].extend(lower_features_mapping[feature])
            self.save_metadata("evaluator_features_mappings", evaluator_features_mappings)
            results_df = combined_df
        elif data_handling.get('use_evaluator', True):
            mandatory_features = pd.DataFrame()
            for key in mandatory_categorical_features:
                for item in mandatory_categorical_features[key]:
                    mandatory_features[item['name']] = \
                        dummies_data_df[key + evaluator_features_sep + item['categories'][0]]
                    for col in item['categories']:
                        mandatory_features[item['name']] = \
                            mandatory_features[item['name']] | dummies_data_df[key + evaluator_features_sep + col]

            results_df = mandatory_features
        else:
            results_df = data_df

        if action_for_continuous_features == self.AUTO_BIN or action_for_continuous_features == self.REMOVE:
            pass
        elif action_for_continuous_features == self.KEEP_AS_IS:
            data_df.drop(non_continuous_columns, axis=1, inplace=True)
            if isinstance(results_df, pd.DataFrame):
                results_df = pd.concat([data_df, results_df], axis=1)
            else:
                results_df = data_df

        # ------------------------- merge final features with y value -------------------------
        print("------------------------- merge final features with y value -------------------------")
        results_df = results_df.join(final_y.rename(
            self.model_settings.get("variable_to_predict", self.data_settings.get('variable_to_predict'))))
        return results_df

