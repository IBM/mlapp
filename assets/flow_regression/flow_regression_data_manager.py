from mlapp.managers import DataManager
from mlapp.utils import pipeline
from mlapp.utils.exceptions.base_exceptions import DataManagerException
import pandas as pd
import numpy as np


class FlowRegressionDataManager(DataManager):
    @pipeline
    def load_train_data(self,*args):
        print(args)

        return
    @pipeline
    def clean_train_data(self, data):
        return data

    @pipeline
    def transform_train_data(self, data):

        return data

    @pipeline
    def load_forecast_data(self, *args):
        try:
            models_outputs = args[0]
        except Exception as e:
            raise DataManagerException("Error: No data input from flow")

        # read data for forecasting
        features = {}
        data = pd.DataFrame()
        df_name = self.data_settings.get('flow_return_data')
        for model_output in models_outputs:
            index = self.data_settings.get('data_index')
            res = model_output.get(df_name, pd.DataFrame(columns=index))
            if len(res) > 0:
                df = res[0]
                # df.set_index(self.data_settings.get('data_index'))
                data = pd.concat([data, df])
            else:
                continue
        for feature_name in self.data_settings.get('flow_return_features'):
            features[feature_name] = []
            for model_output in models_outputs:
                feature_data = model_output.get(feature_name, pd.DataFrame(columns=index))
                if len(feature_data)>0:
                    if isinstance(feature_data, pd.DataFrame):
                        if not isinstance(features[feature_name], pd.DataFrame):
                            features[feature_name]=pd.DataFrame()
                        features[feature_name] = pd.concat([feature_data, features[feature_name]])
                    else:
                        # assume it is a list
                        features[feature_name].append(feature_data)
                else:
                    continue

        for key in features:
            self.save_metadata(key, features[key])

        return data

    @pipeline
    def clean_forecast_data(self, data):
        return data

    @pipeline
    def transform_forecast_data(self, data):
        index = self.data_settings.get('data_index')
        data.reset_index(inplace=True)
        agg_function = self.data_settings['data_handling'].get('agg_function_dataframe')
        values = self.data_settings['data_handling'].get('agg_on_columns')
        predictions_pivoted = pd.pivot_table(data, index=index, values=values, aggfunc=eval(agg_function))

        return predictions_pivoted

    @pipeline
    def load_target_data(self, *args):
        raise NotImplementedError()
