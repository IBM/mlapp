import numpy as np
from pyspark.ml.feature import VectorAssembler
from mlapp.managers import ModelManager, pipeline
from mlapp.utils.automl import AutoMLSpark


class SparkRegressionModelManager(ModelManager):
    @pipeline
    def train_model(self, data):
        # preparation for train
        variable_to_predict = self.model_settings['variable_to_predict']

        # split the data to test and train
        train_percent = self.model_settings.get('train_percent', 0.8)
        train_data, test_data = data.randomSplit([train_percent, (1 - train_percent)])

        # regression
        result = AutoMLSpark('linear').run(train_data, test_data, variable_to_predict, cv=3)

        # store results
        self.save_automl_result(result, obj_type='pyspark')

    @pipeline
    def forecast(self, data_df):
        result = self.get_automl_result()

        # get best model
        model = result.get_best_model()

        # get selected features from feature selection algorithm
        selected_features_names = result.get_selected_features()

        # filter data according to selected features
        filtered_data = data_df.select(selected_features_names)

        # assemble data for train
        assembler = VectorAssembler(inputCols=selected_features_names, outputCol='features')
        data_df = assembler.transform(filtered_data).select('features')

        # prediction
        predictions = model.transform(data_df)
        num_of_predictions = predictions.count()

        # adding prediction to results
        self.add_predictions(np.arange(num_of_predictions),                                 # index array
                             predictions.select('prediction').toPandas()['prediction'],     # y_hat
                             np.array([np.nan] * num_of_predictions),                       # y_true (empty array)
                             'forecast')                                                    # forecast type

    @pipeline
    def refit(self, data):
        raise NotImplementedError()
