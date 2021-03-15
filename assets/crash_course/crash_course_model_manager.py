from mlapp.managers import ModelManager, pipeline
from sklearn.model_selection import train_test_split
from mlapp.utils.automl import AutoMLPandas


class CrashCourseModelManager(ModelManager):
    @pipeline
    def train_model(self, data):
        # prepare data for train
        X, y = data.drop(self.model_settings['target'], axis=1), data[self.model_settings['target']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.model_settings.get('train_percent'))

        # run auto ml
        result = AutoMLPandas('multi_class', **self.model_settings.get('auto_ml', {}))\
            .run(X_train, y_train, X_test, y_test)

        # print report
        result.print_report(full=False)

        # save results
        self.save_automl_result(result)

    @pipeline
    def forecast(self, data):
        # load the model
        model = self.get_automl_result().get_best_model()

        # predict
        predictions = model.predict(data)

        # store the predictions
        self.add_predictions(data.index,  predictions, [], prediction_type='forecast')

    @pipeline
    def refit(self, data):
        raise NotImplementedError()
