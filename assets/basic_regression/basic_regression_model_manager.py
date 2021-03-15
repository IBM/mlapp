from sklearn.model_selection import train_test_split
from mlapp.managers import ModelManager, pipeline
from mlapp.utils.automl import AutoMLPandas


class BasicRegressionModelManager(ModelManager):
    def __init__(self, *args, **kwargs):
        ModelManager.__init__(self, *args, **kwargs)

    @pipeline
    def train_model(self, data):
        variable_to_predict = self.model_settings.get('variable_to_predict', 'target')
        train_size = self.model_settings.get('train_percent', 0.8)

        # preparation for train
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(variable_to_predict, axis=1), data[variable_to_predict], train_size=train_size)

        # auto ml
        result = AutoMLPandas('linear').run(X_train, y_train, X_test, y_test)

        # save results
        self.save_automl_result(result)

    @pipeline
    def forecast(self, data):
        result = self.get_automl_result()

        # get best model
        model = result.get_best_model()

        # get selected features from feature selection algorithm
        selected_features_names = result.get_selected_features()

        # filter data according to selected features
        filtered_data = data[selected_features_names]

        # predicting
        predictions = model.predict(filtered_data)

        # store predictions
        self.add_predictions(data.index, predictions, [], 'forecast')

    @pipeline
    def refit(self, data):
        raise NotImplementedError()
