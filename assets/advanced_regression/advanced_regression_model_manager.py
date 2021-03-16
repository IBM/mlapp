from mlapp.managers import ModelManager, pipeline
import pandas as pd
from numpy import random
from sklearn.model_selection import KFold, train_test_split
from mlapp.utils.automl import AutoMLPandas


class AdvancedRegressionModelManager(ModelManager):
    @pipeline
    def train_model(self, data):
        """
        This function trains the model with the various algorithms and selects the best one.
        :param data: pandas DataFrame that contains X, y for training the model
        :return None
        """
        # preparation for train
        variable_to_predict = self.model_settings['variable_to_predict']
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(variable_to_predict, axis=1), data[variable_to_predict],
            train_size=self.model_settings.get('train_percent', 0.8), random_state=random.seed(0))

        # auto ml
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=random.seed(0))
        fs = self.model_settings.get('auto_ml', {}).get('feature_selection')

        # linear models
        linear_result = AutoMLPandas(
            'linear', feature_selection=fs, **self.model_settings.get('auto_ml', {}).get('estimators', {}).get(
                'linear', {})).run(X_train, y_train, X_test, y_test, cv=cv_splitter)

        # non linear models
        non_linear_result = AutoMLPandas(
            'non_linear', feature_selection=fs, **self.model_settings.get('auto_ml', {}).get('estimators', {}).get(
                'non_linear', {})).run(X_train, y_train, X_test, y_test, cv=cv_splitter)

        # merge results
        final_result = linear_result
        final_result.merge(non_linear_result)
        
        # adding best model
        self.save_automl_result(final_result)

        # adding predictions
        self.add_predictions(X_train.index, final_result.get_train_predictions(), y_train, 'TRAIN')
        self.add_predictions(X_test.index, final_result.get_test_predictions(), y_test, 'TEST')

    @pipeline
    def forecast(self, data_df):
        result = self.get_automl_result()

        # get best model
        model = result.get_best_model()

        # get selected features from feature selection algorithm
        selected_features_names = result.get_selected_features()

        # filter data according to selected features
        filtered_data = data_df[selected_features_names]

        # predicting
        predictions = model.predict(filtered_data)
        predictions_df = pd.DataFrame(data=predictions, columns=['y_hat'], index=filtered_data.index)
        predictions_df['y_true'] = None      # consistency with estimations (possibly add in the actual in the future)
        predictions_df['type'] = 3    # type 3 = forecast
        self.add_predictions(filtered_data.index, predictions, [], 'forecast')

    @pipeline
    def refit(self, data):
        raise NotImplementedError()


