from mlapp.managers import ModelManager
from mlapp.utils import pipeline


class FlowRegressionModelManager(ModelManager):
    @pipeline
    def train_model(self, data):
        pass

    @pipeline
    def forecast(self, data_df):
        self.add_predictions(data_df.index, data_df['y_hat'], None, 'FORECAST')

    @pipeline
    def refit(self, data):
        raise NotImplementedError()
