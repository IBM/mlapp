import copy
import datetime
import collections
import json
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from mlapp.config import settings
from mlapp.utils.automl import AutoMLResults
from mlapp.managers.pipeline_manager import pipeline
from mlapp.managers.io_manager import IOManager
from mlapp.utils.exceptions.framework_exceptions import UnsupportedFileType, MissingConnectionException
from mlapp.handlers.wrappers.database_wrapper import database_instance


class _UserManager(object):
    prediction_types = {
        'TRAIN': 1,
        'TEST': 2,
        'FORECAST': 3
    }

    def __init__(self, config, _input: IOManager, _output: IOManager, run_id: str, *args, **kwargs):
        self.local_storage_path = settings.get("local_storage_path", "output")
        self.run_id = run_id
        self.data_settings = config.get('data_settings', {})
        self.model_settings = config.get('model_settings', {})
        self.job_settings = config.get('job_settings', {})
        self.flow_settings = config.get('flow_settings', {})
        self._input_manager = _input
        self._output_manager = _output

    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_manager_type(self):
        pass

    def _get_all_metadata(self):
        """
        returns all key value pairs saved to metadata.
        :return: dictionary
        """
        return self._input_manager.get_metadata()

    def _get_all_objects(self):
        """
        Returns all saved objects.
        :return: dictionary of objects.
        """
        return self._input_manager.get_objects()

    def save_metadata(self, key, value):
        """
        Saves a metadata value to storage.
        :param key: String.
        :param value: metadata includes all ""json serializable"" objects
        (i.e string, int, dictionaries, list and tuples)
        :return: None
        """
        self._output_manager.set_analysis_metadata_value(self._get_manager_type(), key, copy.deepcopy(value))

    def get_metadata(self, key, default_value=None):
        """
        returns metadata value given a key.
        :param key: String.
        :param default_value: any object, default is None.
        :return: metadata value.
        """
        metadata = self._input_manager.get_metadata_value(self._get_manager_type(), default_value)
        return metadata.get(key, default_value)

    def save_object(self, obj_name, obj_value, obj_type='pkl'):
        """
        Saves objects to storage.
        :param obj_name: String.
        :param obj_value: Obj.
        :param obj_type: String. One of MLApp supported file types: 'pkl', 'pyspark', 'tensorflow', 'keras', 'pytorch'.
        Default='pkl'.
        :return: None
        """
        supported_types = ['pkl', 'pyspark', 'tensorflow', 'keras', 'pytorch']

        if obj_type not in supported_types:
            raise UnsupportedFileType('Unsupported file type: ' + obj_type)
        obj = collections.defaultdict(dict)
        obj[self._get_manager_type()][obj_name] = obj_value     # no deep copy as obj value can be non python
        self._output_manager.add_objects(obj_type, dict(obj))

    def get_object(self, obj_name, default_value=None):
        """
        Returns an object given a key.
        :param obj_name: String.
        :param default_value: String.
        :return: Object
        """
        return self._input_manager.get_objects_value(self._get_manager_type(), {}).get(obj_name, default_value)

    def save_images(self, images):
        """
        Saves images to storage.
        :param images: dictionary of matplotlib/pyplot figures. Keys will be used as image names.
        :return: None.
        """
        self._output_manager.add_images(images)

    def save_image(self, image_name, image):
        """
        Saves image to storage.
        :param image_name: String.
        :param image: matplotlib/pyplot figure.
        :return: None
        """
        self._output_manager.add_image(image_name, image)

    def get_dataframe(self, key):
        """
        Returns the Data Frame by key name
        :return: df
        """
        return self._input_manager.get_dataframe(key)

    def save_dataframe(self, key, value, to_table=None):
        """
        Save a data frame to storage.
        :param key: String
        :param value: a DataFrame.
        :param to_table: String. Database table name. default: None.
        :return: None
        """
        self._output_manager.add_dataframe(key, value, to_table)

    def save_automl_result(self, results: AutoMLResults, obj_type='pkl'):
        """
        Saves an AutoMLResults object.
        :param results: AutoMLResults object
        :param obj_type: type of framework, Default: 'pkl'.
        Supports: 'pkl', 'pyspark', 'tensorflow', 'keras', 'pytorch'.
        :return: None
        """
        self.save_object('model', results.get_best_model(), obj_type=obj_type)
        self.save_images(results.get_figures())
        metadata = results.get_metadata()
        for key in results.get_metadata():
            self.save_metadata(key, metadata[key])

    def get_automl_result(self) -> AutoMLResults:
        """
        Returns an AutoMLResult object that was saved in a previous run
        :return: AutoMLResult object
        """
        result = AutoMLResults()
        result.best_model = self.get_object('model')
        result.best_model_metrics = self.get_metadata('scores', {})
        result.best_estimator = self.get_metadata('estimator_family')
        result.best_model_key = self.get_metadata('model_class_name')
        result.selected_features = self.get_metadata('selected_features_names', [])
        result.intercept = self.get_metadata('intercept')
        result.coefficients = self.get_metadata('coefficients', [])
        result.feature_selection = self.get_metadata('feature_selection')
        result.best_cv_score = self.get_metadata('cv_score')
        return result


class DataManager(_UserManager):
    def _get_manager_type(self):
        return 'data'

    @abstractmethod
    def load_train_data(self, *args):
        """
        Write your own logic to load your train data.
        :return: data object to be passed to next pipeline step
        """
        pass

    @abstractmethod
    def load_forecast_data(self, *args):
        """
        Write your own logic to load your forecast data.
        :return: data object to be passed to next pipeline step
        """
        pass

    @abstractmethod
    def clean_train_data(self, data):
        """
        Write your own logic to clean your train data.
        :param data: data object received from previous pipeline step
        :return: data object to be passed to next pipeline step
        """
        return data

    @abstractmethod
    def clean_forecast_data(self, data):
        """
        Write your own logic to clean your forecast data.
        :param data: data object received from previous pipeline step
        :return: data object to be passed to next pipeline step
        """
        return data

    @abstractmethod
    def transform_train_data(self, data):
        """
        Write your own logic to transform your train data.
        :param data: data object received from previous pipeline step
        :return: data object to be passed to next pipeline step
        """
        return data

    @abstractmethod
    def transform_forecast_data(self, data):
        """
        Write your own logic to transform your forecast data.
        :param data: data object received from previous pipeline step
        :return: data object to be passed to next pipeline step
        """
        return data

    @abstractmethod
    def load_target_data(self, *args):
        """
        Write your own logic to load your historical target actual values (for monitoring purposes).
        """
        pass

    @pipeline
    def cache_features(self, df):
        """
        Used to save the features dataframe. It can later be retrieved using get_features().
        :param df: Dataframe
        :return: None
        """
        self._output_manager.add_dataframe('features', df)

    @pipeline
    def load_features(self):
        """
        Returns the dataframe "features" from input manager.
        :return: df
        """
        return self._input_manager.get_dataframe('features')

    @pipeline
    def update_actuals(self, df):
        if database_instance.empty():
            raise MissingConnectionException('Failed to update actuals. No database configured in environment.')
        database_instance.update_actuals(df)

    def get_input_from_predecessor_job(self, key=None):
        """
        Returns output from predecessor job "features" from input manager (in a flow settings).
        :param key: key name of value
        :return: any object
        """
        if key is not None:
            return self._input_manager.metadata.get('input_from_predecessor', {}).get(key)
        else:
            return self._input_manager.metadata.get('input_from_predecessor', {})


class ModelManager(_UserManager):
    def _get_manager_type(self):
        return 'models'

    @abstractmethod
    def train_model(self, data):
        """
        Write your own logic to train your model.
        :param data: data object received from previous pipeline step
        :return: None
        """
        pass

    #########################################
    #    Should be implemented by child     #
    #########################################
    @abstractmethod
    def forecast(self, data):
        """
        Write your own logic for forecast.
        :param data: data object received from previous pipeline step
        :return: None
        """
        pass

    #########################################
    #    Should be implemented by child     #
    #########################################
    @abstractmethod
    def refit(self, data):
        """
        Write your own logic for refit.
        :param data: data object received from previous pipeline step
        :return: None
        """
        pass

    def add_predictions(self, primary_keys_columns, y_hat, y=pd.Series(dtype='float64'), prediction_type='TRAIN'):
        """
        Creates a prediction dataframe and saves it to storage.
        :param primary_keys_columns: array-like. shared index for y_hat and y_true.
        :param y_hat: Series / array-like. predictions.
        :param y: Series / array-like. True values.
        :param prediction_type: String. (i.e TRAIN, TEST, FORECAST)
        :return: None.
        """
        y_true = None
        prediction_type = prediction_type.upper()
        if y is not None and len(y)>0:
            y_true = y.copy()

        # Check if y_hat has more than 1 dimension:
        if (y_hat is not None) and (len(y_hat.shape) > 1):
            y_hat = y_hat.reshape(1, len(y_hat))[0]

        # If y_true is a dataframe with only one column, use that column:
        if isinstance(y_true, pd.core.frame.DataFrame):
            if len(y_true.columns.to_list()):
                y_true = y_true[y_true.columns[0]]
        try:
            prediction_type_int = self.prediction_types[prediction_type]
        except:
            prediction_type_int = len(self.prediction_types)+1

        new_prediction = pd.DataFrame({
            "y_true": y_true,
            "y_hat": y_hat.copy(),
            "type": np.repeat(prediction_type_int, len(y_hat)),
            "index": primary_keys_columns.copy()
        })

        self._output_manager.add_dataframe('predictions', new_prediction, 'target')

    @pipeline
    def evaluate_prediction_accuracy(self, *args):
        # TODO: add timestamp filter below to query and implement in postgres handler
        data = database_instance.get_model_predictions(model_id=self.job_settings['model_id'])

        from_timestamp = self.model_settings.get('from_timestamp', min(data['timestamp']))
        to_timestamp = self.model_settings.get('to_timestamp', max(data['timestamp']))

        filter_target_df = data[(data['timestamp'] >= from_timestamp) & (data['timestamp'] <= to_timestamp)]

        acc = eval(self.model_settings['model_accuracy_summary'])(
            filter_target_df['y_hat'], filter_target_df['y_true'], filter_target_df['y_hat'],
            filter_target_df['y_true'])

        # we don't need training and testing we just keep one, and remove the name from the key:
        acc = {
            k.split(' ')[1]: acc[k] for k in acc.keys()
            if k.startswith(np.unique([k.split(' ')[0] for k in acc.keys()])[0])
        }

        df = pd.DataFrame({
            'model_id': self.job_settings['model_id'],
            'asset_name': self.job_settings['asset_name'],
            'asset_label_name': self.job_settings.get('asset_label_name'),
            'created_at': datetime.datetime.now(),
            'updated_at': datetime.datetime.now(),
            'timestamp': datetime.datetime.now(),
            'model_accuracy': json.dumps(acc)
        }, index=[0])

        database_instance.insert_df('asset_accuracy_monitoring', df)

