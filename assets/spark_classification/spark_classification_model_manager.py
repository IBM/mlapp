from mlapp.managers import ModelManager, pipeline
import numpy as np
from pyspark.ml.feature import VectorAssembler
from mlapp.utils.automl import AutoMLSpark


class SparkClassificationModelManager(ModelManager):
    @pipeline
    def train_model(self, data):
        # preparation for train
        variable_to_predict = self.model_settings['variable_to_predict']

        # split the data to test and train
        train, test = data.randomSplit([0.9, 0.1], seed=12345)

        # TODO: scale and save scaler
        # scaler = MinMaxScaler(inputCol="features", outputCol='scaledFeatures')
        # scaler_model = scaler.fit(train_vec)
        #
        # train_scaled = scaler_model.transform(train_vec)
        # test_scaled = scaler_model.transform(test_vec)

        # run auto ml
        fs = self.model_settings.get('auto_ml', {}).get('feature_selection')
        result = AutoMLSpark('binary', **self.model_settings.get('auto_ml', {}).get('binary', {}),
                             feature_selection=fs).run(train, test, variable_to_predict, cv=3)

        # print report
        result.print_report()

        # adding objects
        self.save_automl_result(result, obj_type='pyspark')

        # adding predictions
        train_predictions = result.get_train_predictions().select('prediction', variable_to_predict).toPandas()
        test_predictions = result.get_test_predictions().select('prediction', variable_to_predict).toPandas()

        self.add_predictions(train_predictions.index, train_predictions['prediction'],
                             train_predictions[variable_to_predict], 'TRAIN')
        self.add_predictions(test_predictions.index, test_predictions['prediction'],
                             test_predictions[variable_to_predict], 'TEST')

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
        self.add_predictions(
            np.arange(num_of_predictions),                                  # index array
            predictions.select('prediction').toPandas()['prediction'],      # y_hat
            np.array([np.nan] * num_of_predictions),                        # y_true (empty array)
            'forecast')                                                     # prediction type

    @pipeline
    def refit(self, data):
        raise NotImplementedError()
