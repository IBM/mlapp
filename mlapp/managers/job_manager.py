import os
import time
import json
import glob
import datetime
import csv
import logging
import shutil
import sys
import copy
import pandas as pd
import jsonschema
from mlapp.config import settings
from mlapp.managers.pipeline_manager import PipelineManager
from mlapp.utils import general as general_utils
from mlapp.utils.logger import Logger
from mlapp.managers.io_manager import IOManager
from mlapp.handlers.wrappers.database_wrapper import database_instance
from mlapp.handlers.wrappers.file_storage_wrapper import file_storage_instance
from mlapp.handlers.wrappers.spark_wrapper import spark_instance
from mlapp.utils.exceptions.framework_exceptions import SkipToLocalException, UnsupportedFileType, DataFrameNotFound
from mlapp.utils.exceptions.base_exceptions import ConfigKeyError, FrameworkException, ConfigError, JobManagerException
from matplotlib.figure import Figure
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None


class JobManager(object):
    def __init__(self, job_id, config, **kwargs):
        """
        Constructor for he job manager.
        Its main role is to populate the input from external environment (e.g. trained model's objects) and
        store the outputs in the relevant location (e.g. store figures, objects in file storage service or in local
        directory in case a file storage is not defined. Store data in database and more.
        The job manager is using the singleton handlers to connect to the different services like DB and FileStore
        :param job_id: the id of this current job.
        :param config: the config string of user configurations for the asset
        :param kwargs: run_id and any initial input to the asset.
        run_id - All outputs files to be created will be given this identifier
        """

        # Handlers:
        self.output_logger_filename = "output_logs.csv"
        self.file_store_handler = file_storage_instance
        self.db_handler = database_instance
        self.spark_handler = spark_instance

        self.file_objects_types = {
            "pyspark": {
                "type": ("pyspark" if self.file_store_handler.empty() else "pyspark.zip"),
                "load_method": self._load_files_into_spark_object,
                "store_method": self._store_spark_object_into_files
            },
            "pkl": {
                "type": "pkl.pkl",
                "load_method": self._load_pickle_as_object,
                "store_method": self._store_object_as_pickle
            }
            # "tensorflow",
            # "pytorch",
            # "keras",
        }

        # config parts
        self.job_id = job_id

        self.config = config
        self.data_settings = config.get('data_settings', {})
        self.model_settings = config.get('model_settings', {})
        self.flow_settings = config.get('flow_settings', {})
        self.job_settings = config.get('job_settings', {})

        self.filestore_buckets = settings.get("file_store_buckets", {})
        self.local_storage_path = settings.get("local_storage_path", "output")
        self.temporary_storage_path = settings.get("temporary_storage_path", "temporary_output")
        self.temporary_storage_files = set()
        self.set_output_folders()
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.last_time = self.start_time
        self.deploy_environment = settings.get("deploy_environment", "default")
        self.deploy_version = settings.get("deploy_version", "-")

        self.identity = {
            'run_id': kwargs['run_id'],
            'pipeline': self.job_settings.get('pipeline', None),
            'asset_name': self.get_asset_name(kwargs.get('has_flow_summary')),
            'asset_label': self.job_settings.get('asset_label', None)
        }

        # consts
        self.DOT = '.'
        self.LOAD = 'LOAD'
        self.SAVE = 'SAVE'
        self.DATA = 'data'
        self.MODELS = 'models'
        self.FEATURES = 'features'
        self.INPUT_FROM_PREDECESSOR = 'input_from_predecessor'

        self.input_manager = IOManager()
        self.output_manager = IOManager()

    def __del__(self):
        self.clean_output_folders()

    ##########################################################################
    #                                                                        #
    #                            Run Pipeline                                #
    #                                                                        #
    ##########################################################################
    def run_pipeline(self, *args, **kwargs):
        """
        Loads the inputs defined in the config ( e.g. pre trained objects, metadata and more)
        Initiates a pipeline manager, runs it
        Stores the outputs in the relevant locations
        :param args: arguments to be passed to the first stage in the pipeline
        :param kwargs: keyword arguments to be loaded into the Input (an IOManager instance)
        :return: IOManager instance with all the outputs to be stored
        """
        try:
            self._store_job_config()
            self.log_to_file(run_id=None)  # no `run_id` output yet
            self.validate_config()
            self.load_input(**kwargs)
            pipeline_manager = PipelineManager(
                self.identity['run_id'],
                self.job_settings['pipeline'],
                _input=self.input_manager, _output=self.output_manager,
                config=copy.deepcopy(self.config),
                **kwargs
            )
            self.pipeline_start_print(self.identity['run_id'])
            pipeline_manager.run(*args)
            self.pipeline_end_print(self.identity['run_id'])
            self.store_output()
            self.temporary_files_message()
        except Exception as error:
            self.identity['run_id'] = None    # pipeline failed
            log = logging.getLogger(self.job_id)
            log.error(str(error), exc_info=True)
            raise error
        finally:
            # saving logger file
            self._store_logger_file(run_id=self.identity['run_id'])
        return self.output_manager

    ##########################################################################
    #                                                                        #
    #                             Get Functions                              #
    #                                                                        #
    ##########################################################################

    def get_custom_filename(self, run_id, name):
        """
        :param run_id: unique id of the asset
        :param name: file name
        :return: proper concatenation of the file name
        """
        return str(run_id) + '_' + self.identity['asset_name'] + '_' + name

    def get_dataframe_filename(self, run_id, df_name):
        """
        :param run_id: unique id of the asset
        :param df_name: dataframe name
        :return: proper concatenation of the dataframe file name
        """
        return str(run_id) + '_' + self.identity['asset_name'] + self.DOT + df_name + '.csv'

    def get_config_filename(self, run_id):
        """
        :param run_id: unique id of the asset
        :return: proper concatenation of the config file name
        """
        return str(run_id) + '_' + self.identity['asset_name'] + '.config.json'

    def get_job_config_filename(self):
        """
        :return: proper concatenation of the config file name
        """
        return str(self.job_id) + '.config.json'

    def get_metadata_filename(self, run_id):
        """
        :param run_id: unique id of the asset
        :return: proper concatenation of the metadata file name
        """
        return str(run_id) + '_' + self.identity['asset_name'] + '.metadata.json'

    @staticmethod
    def get_flow_metadata_filename(run_id):
        """
        :param run_id:  unique id of the asset
        :return: proper concatenation of the flow-summary-asset metadata file name
        """
        return str(run_id) + '_flow_summary.metadata.json'

    def get_features_filename(self, run_id):
        """
        :param run_id: unique id of the asset
        :return: proper concatenation of the feature file name
        """
        return str(run_id) + '_' + self.identity['asset_name'] + '.features.csv'

    def get_objects_filename(self, run_id, manager, mlapp_type, file_type, model_name=None, class_name=None):
        """
        :param run_id: string, unique id of the asset
        :param manager: string , manager name
        :param mlapp_type: string , one of the keys in self.file_objects_types
        :param file_type: string , file type for example, .csv .txt etc
        :param model_name: string , name of the asset
        :param class_name: string, spark object class name.
        :return: proper concatenation of the object file name
        """
        if model_name is None and class_name is None:
            return str(run_id) + '_' + self.identity['asset_name'] + \
                   self.DOT + manager + self.DOT + mlapp_type + self.DOT + file_type
        elif class_name is None:
            return str(run_id) + '_' + self.identity['asset_name'] + self.DOT + model_name + \
                   self.DOT + manager + self.DOT + mlapp_type + self.DOT + file_type
        elif model_name is None:
            return str(run_id) + '_' + self.identity['asset_name'] + self.DOT + class_name + \
                   self.DOT + manager + self.DOT + mlapp_type + self.DOT + file_type
        else:
            return str(run_id) + '_' + self.identity['asset_name'] + self.DOT + class_name + \
                   self.DOT + model_name + self.DOT + manager + self.DOT + mlapp_type + self.DOT + file_type

    def get_objects_modules_filename(self, run_id):
        """
        :param run_id: string, unique id of the asset
        :return: proper concatenation of the object modules json file name
        """
        return str(run_id) + '_' + self.identity['asset_name'] + '.objects.modules.json'

    def get_logger_filename(self, run_id=None):
        """
        :param run_id: string, unique id of the asset
        :return: proper concatenation of the logger file name
        """
        if run_id is not None:
            return str(run_id) + '_' + str(self.job_id) + '.logger.txt'
        else:
            return str(self.job_id) + '.logger.txt'

    def get_asset_name(self, has_flow_summary):
        """
        :param has_flow_summary: if the config contains flow summary section
        :return: model name from the config
        """
        if has_flow_summary is not None and not has_flow_summary:
            return None
        else:
            asset_name = self.job_settings.get('asset_name', None)
            if asset_name is None:
                #  model_name deprecation
                asset_name = self.job_settings.get('model_name', None)
                if asset_name is None:
                    raise ConfigKeyError('asset_name is required in job_settings config.')
                else:
                    try:
                        raise DeprecationWarning('model_name is deprecated please use asset_name instead.')
                    except Warning as w:
                        print("DeprecationWarning: " + str(w))
            return asset_name

    ###############################################################
    #                                                             #
    #                        Load Input                           #
    #                                                             #
    ###############################################################
    def load_input(self, **kwargs):
        """
          This is Main Load Input function that called in run_pipeline function.
          It loads any input specified in the config (e.g. trained model object, saved features DataFrame etc.).
          The function will load the input into an IOManager instance - the InputManager.
          :param kwargs:  input from predecessor pipeline. Passed by the flow manager.
          :return: None

        :param kwargs:
        :return:
        """

        # Todo: 1. merge in kwargs into the input_manager by calling the update_recursive_dict(input,kwargs)
        if kwargs:
            self.input_manager.add_analysis_metadata(self.INPUT_FROM_PREDECESSOR, kwargs)

        # get ids from config
        config_model_id = self._load_model_id_from_config()
        config_data_id = self._load_data_id_from_config()
        config_reuse_features_id = self._load_reuse_features_id_from_config()
        # config_reuse_features_flag = self._load_reuse_features_flag_from_config()

        # if there is no ids in the config, skip load input method.
        if config_model_id is None and config_data_id is None and config_reuse_features_id is None:
            return

        # loads required files from storage.
        model_config = self.load_config(config_model_id)
        # looking for the right config_data_id
        config_data_id = self._decide_on_config_data_id(config_data_id, model_config, config_model_id)
        data_config = self.load_config(config_data_id)
        train_metadata = self.load_metadata(config_model_id)
        train_features = self.load_features(config_reuse_features_id)
        train_objects = self.load_objects(config_model_id, config_data_id)

        # updates config if necessary
        self.update_config(model_config, data_config)

        # gets train data and models metadata
        if train_metadata is not None:
            train_data_metadata = train_metadata.get(self.DATA, {})
            train_models_metadata = train_metadata.get(self.MODELS, {})

            # add analysis metadata to input manager
            self.input_manager.add_analysis_metadata(self.DATA, train_data_metadata)
            self.input_manager.add_analysis_metadata(self.MODELS, train_models_metadata)

        # gets train data and models objects
        if train_objects is not None:
            train_data_objects = train_objects.get(self.DATA, {})
            train_models_objects = train_objects.get(self.MODELS, {})

            # add objects to input manager
            self.input_manager.add_objects(self.DATA, train_data_objects)
            self.input_manager.add_objects(self.MODELS, train_models_objects)

        # check if features reused
        if train_features is not None:
            self.input_manager.add_dataframe(self.FEATURES, train_features)

    @staticmethod
    def _decide_on_config_data_id(config_data_id, model_config, config_model_id):
        """
        Decide on the right data_id for model forecast to reuse
        scenario 1: data_id is supplied in the main config
        scenario 2: data_id missing in main config, but supplied in the reused model config (the train model config)
        scenario 3: data_id is missing and is missing in the reused model config, in this case data_id = model_id
        @param config_data_id: string
        @param model_config: dict
        @param config_model_id: string
        @return: string
        """
        if config_data_id is None:
            if model_config is None:
                return
            config_model_data_id = model_config.get('task_settings', {}).get('data_id', None)
            if config_model_data_id is None:
                new_config_data_id = config_model_id
            else:
                new_config_data_id = config_model_data_id
        else:
            new_config_data_id = config_data_id
        return new_config_data_id

    def load_config(self, config_run_id):
        """
        loads the config file of a previous asset run from filesystem
        :param config_run_id: the id of the run
        :return:
        """
        if config_run_id is None:
            return
        config_filename = self.get_config_filename(config_run_id)
        config = self._load_json_as_object(config_filename, bucket_name='configs')
        return config

    def load_metadata(self, run_id):
        """
        loads the metadata file of a previous asset run from filesystem
        :param run_id: the id of the run
        :return:
        """
        if run_id is None:
            return
        data_output_filename = self.get_metadata_filename(run_id)
        data_results = self._load_json_as_object(data_output_filename, bucket_name='metadata')
        return data_results

    def load_features_with_flag(self, data_id, reuse_flag=False):
        """
        :param data_id:
        :param reuse_flag:
        :return:
        """
        if not reuse_flag or data_id is None:
            return
        features_filename = self.get_features_filename(data_id)
        features = self._load_csv_as_data_frame(features_filename, bucket_name='csvs')
        return features

    def load_features(self, config_reuse_features_id):
        """
        loads the features file of a previous asset run from filesystem
        :param config_reuse_features_id: the id of the run that created the features
        :return:
        """
        if config_reuse_features_id is None:
            return
        features_filename = self.get_features_filename(config_reuse_features_id)
        features = self._load_csv_as_data_frame(features_filename, bucket_name='csvs')
        return features

    def load_objects(self, model_id, data_id):
        """
        loads the object files of a previous asset run from filesystem
        :param model_id: run_id id of the run that created the model
        :param data_id: run_id id of the run that created the features
        :return: dictionary with objects
        """
        if model_id is None and data_id is None:
            return
        object_results_filenames = self._load_model_objects_files_names(model_id, data_id)
        object_results = {}
        try:
            modules = self._load_objects_modules_file(model_id)
            for f in object_results_filenames:
                nargs = len(f.split('.'))
                if nargs == 4:
                    file_name, manager, mlapp_type, file_type = f.split('.')
                    object_results[manager] = \
                        self.file_objects_types.get(mlapp_type, self.file_objects_types['pkl'])['load_method'](
                            f, bucket_name='objects')
                else:
                    # first 6 types of names
                    file_name, class_name, model_name, manager, mlapp_type, file_type = f.split('.')[:6]
                    import_module = self.import_object_from_module(module=modules.get(class_name, None),
                                                                   class_name=class_name)
                    if manager not in object_results.keys():
                        object_results[manager] = {}
                    object_results[manager][model_name] = \
                        self.file_objects_types.get(mlapp_type, self.file_objects_types['pkl'])['load_method'](
                            f, bucket_name='objects', module=import_module)

            return object_results
        except Exception as e:
            print("An error accord while load objects. Reason: " + str(e))
            raise FrameworkException()

    ##################################################################
    #                                                                #
    #                       Load Private Functions                   #
    #                                                                #
    ##################################################################

    def _load_model_id_from_config(self):
        model_id = self.job_settings.get('model_id', None)
        return model_id

    def _load_data_id_from_config(self):
        data_id = self.job_settings.get('data_id', None)
        return data_id

    def _load_reuse_features_id_from_config(self):
        reuse_features_id = self.job_settings.get('reuse_features_id', None)
        return reuse_features_id

    def _load_reuse_features_flag_from_config(self):
        reuse_features_flag = self.job_settings.get('reuse_features', False)
        return reuse_features_flag

    def _load_pickle_as_object(self, file_name, module=None, bucket_name='objects'):
        try:
            if self.file_store_handler.empty() or self.filestore_buckets.get(bucket_name) is None:
                raise SkipToLocalException()

            local_path = self._load_file_from_file_store(self.filestore_buckets[bucket_name], file_name)
        except SkipToLocalException:
            local_path = os.path.join(self.local_storage_path, file_name)
        except Exception as e:
            print("Failed to load object from file storage. Reason: " + str(e))
            raise FrameworkException()

        return general_utils.load_pickle_to_object(local_path)

    def _load_files_into_spark_object(self, file_name, module=None, bucket_name='objects'):
        try:
            if module is None:
                raise JobManagerException('spark module object path must be passed in order to load a spark model.')

            if self.file_store_handler.empty():
                raise SkipToLocalException()

            file_path = self._load_file_from_file_store(self.filestore_buckets[bucket_name], file_name)
            extract_dir = os.path.splitext(file_name)[0]  # rm the word zip
            if self.file_store_handler.empty():
                general_utils.uncompress_zip_to_folder(
                    file_path, extract_dir=os.path.join(self.local_storage_path, extract_dir))
            else:
                general_utils.uncompress_zip_to_folder(
                    file_path, extract_dir=os.path.join(self.temporary_storage_path, extract_dir))
        except SkipToLocalException:
            try:
                file_path = os.path.join(self.local_storage_path, file_name)
            except Exception as e:
                print("spark model file was not loaded. Reason: " + str(e))
                return None
        except Exception as e:
            print("Failed to load spark object from file storage. Reason: " + str(e))
            raise FrameworkException()

        return self.spark_handler.load_model(file_path=file_path.replace('.zip', ''), module=module)

    def _load_objects_modules_file(self, model_id):
        try:
            if model_id is None:
                return
            modules_filename = self.get_objects_modules_filename(model_id)
            modules = self._load_json_as_object(modules_filename, bucket_name='objects')
            return modules
        except:
            return {}

    def _load_model_objects_files_names(self, model_id, data_id):
        files_path = []
        try:
            types_suffix = list(map(lambda s: s.get('type', 'pkl'), self.file_objects_types.values()))

            if self.file_store_handler.empty():
                raise SkipToLocalException()

            if model_id is not None:
                files_path.extend(map(lambda x: x.decode("utf-8") if hasattr(x, 'decode') else x,
                                      self._load_files_names_from_file_storage(model_id)))

            if data_id is not None:
                files_path.extend(map(lambda x: x.decode("utf-8") if hasattr(x, 'decode') else x,
                                      self._load_files_names_from_file_storage(data_id)))

            files_path = filter(lambda f: f.endswith(tuple(types_suffix)), files_path)

        except SkipToLocalException:
            if model_id is not None:
                files_path.extend(self._load_files_names_from_local_storage(model_id))

            if data_id is not None:
                files_path.extend(self._load_files_names_from_local_storage(data_id))

            files_path = filter(lambda f: f.endswith(tuple(types_suffix)), files_path)
        except Exception as e:
            raise FrameworkException('Failed to load objects files from file storage. reason: ' + str(e))

        return files_path

    def _load_files_names_from_local_storage(self, run_id):
        # reads all files starting with id as prefix local storage.
        return list(map(lambda f: os.path.basename(f),
                        glob.glob(os.path.join(self.local_storage_path, run_id + "_*"))))

    def _load_files_names_from_file_storage(self, run_id):
        # reads all files starting with id as prefix from file storage.
        return self.file_store_handler.list_files(
            bucket_name=self.filestore_buckets.get('objects'), prefix=run_id, **self.identity)

    def _load_csv_as_data_frame(self, file_name, bucket_name='csvs'):
        try:
            if self.file_store_handler.empty() or self.filestore_buckets.get(bucket_name) is None:
                raise SkipToLocalException()

            local_file_path = self._load_file_from_file_store(self.filestore_buckets[bucket_name], file_name)
            df = pd.read_csv(local_file_path)

        except SkipToLocalException:
            try:
                df = pd.read_csv(os.path.join(self.local_storage_path, file_name))
            except Exception as e:
                print("File %s was not loaded. Reason: %s" % (str(file_name), str(e)))
                raise DataFrameNotFound(
                    "Error: file " + file_name + " is not found. Make sure you provided the right names and ids.")
        except Exception as e:
            print("File %s was not loaded. Reason: %s" % (str(file_name), str(e)))
            raise FrameworkException()
        return df

    def _load_json_as_object(self, file_name, bucket_name=None):
        try:
            if self.file_store_handler.empty() or self.filestore_buckets.get(bucket_name) is None:
                raise SkipToLocalException()

            local_path = self._load_file_from_file_store(self.filestore_buckets[bucket_name], file_name)
        except SkipToLocalException:
            local_path = os.path.join(self.local_storage_path, file_name)
        except Exception as e:
            raise FrameworkException(str(e))
        return general_utils.read_json_file(local_path)

    def _load_file_from_file_store(self, bucket_name, filename):
        unicode_file_name = str(filename)
        try:
            if self.file_store_handler.empty():
                raise SkipToLocalException()
            try:
                # trying to get file from temporary_storage_path
                to_path = os.path.join(self.temporary_storage_path, unicode_file_name)
                self.file_store_handler.download_file(bucket_name, unicode_file_name, to_path, **self.identity)
            except Exception:
                try:
                    # trying to get file from local_storage_path
                    to_path = os.path.join(self.local_storage_path, unicode_file_name)
                    self.file_store_handler.download_file(bucket_name, unicode_file_name, to_path, **self.identity)
                except Exception as e:
                    raise FrameworkException(str(e))
            return to_path
        except SkipToLocalException as e:
            raise FrameworkException(str(e))

    ###############################################################
    #                                                             #
    #                Store Output Main Functions                  #
    #                                                             #
    ###############################################################

    def store_output(self):
        """
        This is Main store output function that is called in run_pipeline function
        It goes through the different structures in OutputManager of the asset and stores them
        :return: None
        """
        # call store methods
        self.store_configs()
        self.store_metadata()
        self.store_dataframes()
        self.store_objects()
        self.store_images()

    def store_dataframes(self):
        """
        Stores the DataFrame/s saved by the data scientist in the outputManager
        Iterates over dataframes and stores each DataFrame according to its key
        :return: None
        """
        # getting dataframes from output manager
        dataframes = self.output_manager.get_dataframes()
        tables = self.output_manager.get_tables()

        # iterate over dataframes and storing each dataframe according to key
        # if to table is not None or empty string, it will try to store the df in DB.
        for df_name, df in dataframes.items():
            if df is None:
                print("Warning: %s Dataframe not stored" % (str(df_name)))
            table_name = tables.get(df_name)
            filename = self.get_dataframe_filename(self.identity['run_id'], df_name)
            self._store_dataframe(df, filename, run_id=self.identity['run_id'], table_name=table_name)

    def store_objects(self):
        """
         Stores the objects saved by the data scientist in the outputManager
         Iterates over file types saved in output manager and store each object according to its type
        :return: None
        """
        # getting object from output manager
        objects = self.output_manager.get_objects()

        # iterate over file types saved in output manager and storing each object according to its type
        for file_type, content in objects.items():

            # gets mlapp relevant type dictionary
            obj_dict = self.file_objects_types.get(file_type, self.file_objects_types['pkl'])

            # handle model manager storing objects
            models_content = content.get(self.MODELS, {})
            if models_content:
                # gets mlapp_file_type
                try:
                    mlapp_type, _ = obj_dict['type'].split(self.DOT)
                except:
                    mlapp_type = obj_dict['type']

                if isinstance(models_content, dict) and mlapp_type == 'pyspark':
                    objects_modules = {}
                    for model_name, model_object in models_content.items():
                        # gets models file name according to file type
                        model_filename = self.get_objects_filename(self.identity['run_id'], "models", mlapp_type,
                                                                   file_type, model_name=model_name,
                                                                   class_name=model_object.__class__.__name__)
                        # call store method according to file type
                        obj_dict['store_method'](model_object, model_filename, bucket_name='objects')

                        # save object module for loading
                        objects_modules[model_object.__class__.__name__] = model_object.__module__

                    # stores objects modules in a json file
                    self._store_object_modules(objects_modules, bucket_name='objects')

                else:
                    # gets models file name according to file type
                    models_filename = self.get_objects_filename(
                        self.identity['run_id'], "models", mlapp_type, file_type, model_name=None)

                    # call store method according to file type
                    obj_dict['store_method'](models_content, models_filename, bucket_name='objects')

            # handle data manager storing objects
            data_content = content.get(self.DATA, {})
            if data_content:
                # gets mlapp_type
                mlapp_type, _ = obj_dict['type'].split(self.DOT)

                # gets data file name according to file type
                data_filename = self.get_objects_filename(self.identity['run_id'], "data", mlapp_type, file_type)

                # call store method according to file type
                obj_dict['store_method'](data_content, data_filename, bucket_name='objects')

    def store_metadata(self):
        """
        This function stores the model metadata either in the database, or in a file storage as a JSON,
        or in the CSV logger file
        :return: None
        """
        run_id = self.identity['run_id']
        file_name = self.get_metadata_filename(run_id)
        metadata = self.output_manager.get_metadata()
        json_validation_warning_message = "%s is not json serializable, removed from metadata."
        general_utils.validate_json(metadata, json_validation_warning_message)

        try:
            if self.db_handler.empty():
                raise SkipToLocalException("No database set up in 'config.py'.")

            # trying to save metadata in db
            self._store_metadata_in_db(run_id, metadata=metadata)
            print("Stored the metadata in the db.")
        except SkipToLocalException:
            pass
        except Exception as e:
            print("Failed to store metadata in db. Reason: " + str(e))

        saved_path = None
        try:
            saved_path = self._store_object_as_json(metadata, file_name, bucket_name='metadata')
        except Exception as e:
            print("Error: in storing metadata please check if your metadata is a valid JSON. Reason: " + str(e))

        try:
            if os.path.exists(saved_path):
                self._store_object_in_csv_logger(run_id, metadata)
            print("Saved metadata in the local CSV logger file.")
        except Exception as e:
            print("Failed to store metadata in CSV logger file. Reason: " + str(e))

    def store_flow_metadata(self, flow_id, **kwargs):
        """
        This function stores the flow_job_manager metadata and relevant only for the flow summary job manager.
        It stores the flow metadata either in the database, or in a file storage as a JSON, or in the CSV logger file
        :param flow_id: string
        :return: None
        """
        metadata = {}
        pipelines_metadata = kwargs.get('pipelines_metadata', {})
        metadata['flow_id'] = flow_id
        metadata['pipelines_metadata'] = pipelines_metadata
        json_validation_warninig_message = "%s is not json serializable, removed from metadata."
        general_utils.validate_json(metadata, json_validation_warninig_message)
        file_name = self.get_flow_metadata_filename(flow_id)

        # trying to save analysis results in db
        try:
            if self.db_handler.empty():
                raise SkipToLocalException("No database set up in 'config.py'.")

            self._store_flow_metadata_in_db(flow_id, metadata=metadata, properties={})
            print("Stored the metadata in the db.")
        except SkipToLocalException:
            pass
        except Exception as e:
            print("Failed to store metadata in db. Reason: " + str(e))

        saved_path = None
        try:
            saved_path = self._store_object_as_json(metadata, file_name, bucket_name='metadata')
        except Exception as e:
            print("Error: in storing metadata please check if your metadata is a valid JSON. Reason: " + str(e))

        try:
            if os.path.exists(saved_path):
                self._store_object_in_csv_logger(flow_id, metadata)
                print("Stored the metadata in the local CSV logger file.")
        except Exception as e:
            print("Failed to store metadata in CSV logger file. Reason: " + str(e))

    def store_configs(self):
        """
        Stores the config of this mlapp run as json file. Saves in Filestore if defined or locally.
        :return: None
        """
        model_config = self.config.copy()
        file_name = self.get_config_filename(self.identity['run_id'])
        self._store_object_as_json(model_config, file_name, bucket_name='configs')

    def store_images(self, bucket_name='imgs'):
        """
        Stores the images saved by the data scientist in the outputManager. Saves in Filestore if defined or locally.
        :param bucket_name: dedicated bucket name in the FileStore
        :return: None
        """
        images = self.output_manager.get_images_files()
        images_for_upload = []

        # saving images in local storage and preparing them for upload
        model_filename = self.get_custom_filename(self.identity['run_id'], '')
        for image_name, figure in images.items():
            if go and isinstance(figure, go.Figure):
                file_name = model_filename + image_name + '.html'
            elif isinstance(figure, Figure):
                file_name = model_filename + image_name + '.png'
            else:
                raise UnsupportedFileType(
                    f"{image_name} is not in the right format. "
                    f"ML App supports only figures from the matplotlib or plotly.js library.")

            if self.file_store_handler.empty():
                file_path = os.path.join(self.local_storage_path, file_name)
            else:
                file_path = os.path.join(self.temporary_storage_path, file_name)
            try:
                if go and isinstance(figure, go.Figure):
                    figure.write_html(file_path, include_plotlyjs='directory')
                else:
                    general_utils.save_figure_to_png(figure, file_path, file_name)
                images_for_upload.append({
                    'file_name': file_name,
                    'file_path': file_path
                })
            except Exception as e:
                print(e)
        # saving images in file storage
        if images_for_upload:
            try:
                if self.file_store_handler.empty() or self.filestore_buckets.get(bucket_name) is None:
                    raise SkipToLocalException("No file storage set up in 'config.py'")
                for image in images_for_upload:
                    self._store_file_in_file_store(self.filestore_buckets['imgs'], image['file_name'],
                                                   image['file_path'])
                    os.remove(image['file_path'])
            except SkipToLocalException:
                pass
            except Exception as e:
                # Keeping images in local file store in case of failure
                print("Failed to store images in file storage. Reason: " + str(e))
                print("Stored images in local file storage instead.")

    ###################################################################
    #                                                                 #
    #                       Store Private Functions                   #
    #                                                                 #
    ###################################################################

    def _store_job_config(self):
        job_config = self.config.copy()
        file_name = self.get_job_config_filename()
        self._store_object_as_json(job_config, file_name, bucket_name='configs')

    def _store_metadata_in_db(self, run_id, metadata, properties=None, table='analysis_results'):
        """
        Storing meta in db
        :param run_id: id of the current running pipeline
        :param metadata: analysis results object to store in db
        :return: None
        """
        if properties is None:
            properties = {}

        model_properties = {'job_id': self.job_id, 'deploy_version': self.deploy_version}
        if self.job_settings.get('asset_label', False):
            model_properties['asset_label'] = self.job_settings['asset_label']
        if self.job_settings.get('batch_name', False):
            model_properties['batch_name'] = self.job_settings['batch_name']
        if properties:
            for prop, content in properties.items():
                model_properties[prop] = content

        config_str = json.dumps(model_properties)
        metadata = json.dumps(metadata, cls=general_utils.NumpyEncoder)

        if isinstance(self.identity['pipeline'], str):
            curr_pipeline = self.identity['pipeline']
        else:
            curr_pipeline = ','.join(self.identity['pipeline'])

        data_to_insert = {
            'model_id': run_id,
            'asset_name': self.identity['asset_name'],
            'pipeline': curr_pipeline,
            'properties': config_str,
            'metadata': metadata,
            'environment': self.deploy_environment,
            'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if self.job_settings.get('asset_label', False):
            data_to_insert['asset_label'] = self.job_settings['asset_label']

        df_for_insert = pd.DataFrame(data=[data_to_insert])

        self.db_handler.insert_df(table, df_for_insert)

    def _store_flow_metadata_in_db(self, flow_id, metadata, properties=None, table='flows'):
        """
        Storing meta in db
        :param flow_id: id of the current running model
        :param metadata: analysis results object to store in db
        :return: None
        """
        if properties is None:
            properties = {}

        model_properties = {'job_id': self.job_id}
        if self.job_settings.get('asset_label', False):
            model_properties['asset_label'] = self.job_settings['asset_label']
        if self.job_settings.get('batch_name', False):
            model_properties['batch_name'] = self.job_settings['batch_name']
        if properties:
            for prop, content in properties.items():
                model_properties[prop] = content

        config_str = json.dumps(model_properties)
        metadata = json.dumps(metadata, cls=general_utils.NumpyEncoder)

        df_for_insert = pd.DataFrame(data=[{
            'flow_id': flow_id,
            'metadata': metadata,
            'properties': config_str,
            'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])

        self.db_handler.insert_df(table, df_for_insert)

    def _store_logger_file(self, run_id=None, bucket_name='logs'):
        log = logging.getLogger(self.job_id)
        for handler in log.handlers:
            handler.close()
            log.removeHandler(handler)
        sys.stdout = sys.__stdout__

        try:
            if self.file_store_handler.empty():
                raise SkipToLocalException("No file storage set up in 'config.py'")
            logger_file_name = self.get_logger_filename()
            fallback_path = os.path.join(self.temporary_storage_path, logger_file_name)
            if os.path.exists(fallback_path):
                if run_id is not None:
                    logger_file_name = self.get_logger_filename(run_id)
                    new_fallback_path = os.path.join(self.temporary_storage_path, logger_file_name)
                    shutil.move(fallback_path, new_fallback_path)
                    fallback_path = new_fallback_path
                try:
                    self._store_file_in_file_store(
                        self.filestore_buckets[bucket_name], logger_file_name, fallback_path)
                    os.remove(fallback_path)
                    print("Stored logger file %s in the file storage." % str(logger_file_name))
                except Exception as e:
                    print("Failed to store logger file %s in the file storage. Reason: %s" %
                          (str(logger_file_name), str(e)))
                    print("Logger file saved in %s storage path instead." % str(self.temporary_storage_path))
                    self.temporary_storage_files.add(str(logger_file_name))
        except SkipToLocalException:
            logger_file_name = self.get_logger_filename()
            local_path = os.path.join(self.local_storage_path, logger_file_name)
            if os.path.exists(local_path):

                if run_id is not None:
                    logger_file_name = self.get_logger_filename(run_id)
                    new_local_path = os.path.join(self.local_storage_path, logger_file_name)
                    os.rename(local_path, new_local_path)
                    print("Stored logger file %s in the local storage." % str(logger_file_name))
        except Exception as e:
            print('An Error accord when trying to write in to logger file. Reason: %s' % str(e))

    def _store_object_as_pickle(self, obj, file_name, bucket_name='objects'):
        # Trying to save model pickle in file store
        try:
            if self.file_store_handler.empty():
                raise SkipToLocalException("No file storage set up in 'config.py'")

            file_path = os.path.join(self.temporary_storage_path, file_name)
            general_utils.save_object_to_pickle(obj, file_path)
            self._store_file_in_file_store(self.filestore_buckets[bucket_name], file_name, file_path)
            os.remove(file_path)
        except SkipToLocalException:
            file_path = os.path.join(self.local_storage_path, file_name)
            general_utils.save_object_to_pickle(obj, file_path)
            print("Stored model in file %s in local storage %s" % (str(file_name), str(self.local_storage_path)))
        except Exception as e:
            print("Failed to store model pickle in file storage. Reason: " + str(e))
            print("Model is saved in %s storage path instead." % str(self.temporary_storage_path))
            self.temporary_storage_files.add(str(file_name))

    def _store_file_in_file_store(self, bucket_name, file_name, file_local_path):
        self.file_store_handler.upload_file(bucket_name, file_name, file_local_path, **self.identity)

    def _store_object_as_json(self, obj, file_name, bucket_name=None):
        saved_path = os.path.join(self.temporary_storage_path, file_name)
        try:
            if self.file_store_handler.empty():
                raise SkipToLocalException("No file storage set up in 'config.py'")

            # saving object as json
            general_utils.save_object_as_json(obj, saved_path)

            if self.job_settings.get('compress_files', False):
                general_utils.get_compressed_file_and_remove_uncompressed(saved_path)
                saved_path += ".zip"
                file_name += ".zip"

            # uploading to file store
            self._store_file_in_file_store(self.filestore_buckets[bucket_name], file_name, saved_path)
            os.remove(saved_path)
        except SkipToLocalException:
            # saving locally in local_storage_path
            try:
                saved_path = os.path.join(self.local_storage_path, file_name)

                # saving object as json
                general_utils.save_object_as_json(obj, saved_path)
            except Exception as e:
                print("Failed to store json %s in local storage. Reason: %s" % (str(file_name), str(e)))
                raise FrameworkException()
        except Exception as e:
            print("Failed to store json %s in file storage. Reason: %s" % (str(file_name), str(e)))
            if os.path.exists(saved_path):
                self.temporary_storage_files.add(str(file_name))
                print('Stored json %s in %s' % (str(file_name), str(self.temporary_storage_path)))
                self.temporary_storage_files.add(str(file_name))

        return saved_path

    def _store_object_in_csv_logger(self, run_id, dictionary):
        """
        Storing analysis results in csv logger
        :param run_id: current running run_id id
        :param dictionary: static dictionary
        :return: None
        """
        row = [key + ": " + str(value) for key, value in dictionary.items()]
        row.insert(0, "run_id: " + str(run_id))
        file_path = os.path.join(self.local_storage_path, self.output_logger_filename)
        with open(file_path, 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _store_dataframe(self, df, file_name, run_id=None, dtype=None, bucket_name='csvs', table_name=None):
        # trying to insert data frame to the database
        try:
            if self.db_handler.empty() or table_name is None:
                raise SkipToLocalException("No Database set up in 'config.py'.")

            self._store_dataframe_in_db(df, table_name, run_id=run_id, dtype=dtype)
        except Exception as e:
            if not isinstance(e, SkipToLocalException):
                print("Failed to store Dataframe in database table %s. Reason: %s" % (str(table_name), str(e)))

            self._store_data_frame_as_csv(df, file_name, bucket_name=bucket_name)

    def _store_dataframe_in_db(self, df, table_name, run_id=None, dtype=None):
        config_model_id = self._load_model_id_from_config()

        # TODO: split index into columns and add asset name and asset label to table name
        # adding column of analysis id
        if run_id is not None:
            df['forecast_id'] = run_id

            if config_model_id is None:
                df['model_id'] = run_id
            else:
                df['model_id'] = config_model_id

            df['timestamp'] = pd.datetime.now()

        self.db_handler.insert_df(table_name, df)  # TODO: add dtypes to db handlers

    def _store_data_frame_as_csv(self, df, file_name, bucket_name='csvs'):
        # trying to save predictions in file store
        try:
            if self.file_store_handler.empty():
                raise SkipToLocalException("No file storage set up in 'config.py'.")

            file_path = os.path.join(self.temporary_storage_path, file_name)

            # saving to backup path
            df.to_csv(file_path, index=False)

            self._store_file_in_file_store(self.filestore_buckets[bucket_name], file_name, file_path)
            os.remove(file_path)
        except SkipToLocalException:
            file_path = os.path.join(self.local_storage_path, file_name)
            # saving to local path
            df.to_csv(file_path, index=False)
            print("Stored %s in local storage %s" % (str(file_name), str(self.local_storage_path)))
        except Exception as e:
            print("Failed to store Dataframe file %s. Reason: %s" % (str(file_name), str(e)))
            print("Stored %s in %s" % (str(file_name), str(self.temporary_storage_path)))
            self.temporary_storage_files.add(str(file_name))

    def _store_object_modules(self, data, bucket_name=None):
        file_name = self.get_objects_modules_filename(self.identity['run_id'])
        try:
            if self.file_store_handler.empty():
                raise SkipToLocalException("No file storage set up in 'config.py'")

            file_path = os.path.join(self.temporary_storage_path, file_name)

            if not os.path.exists(file_path):
                general_utils.save_object_as_json(data, file_path)
            else:
                general_utils.update_json_file(data, file_path)

            self._store_file_in_file_store(self.filestore_buckets[bucket_name], file_name, file_path)
            os.remove(file_path)
        except SkipToLocalException:
            # storing_local
            file_path = os.path.join(self.local_storage_path, file_name)

            if not os.path.exists(file_path):
                general_utils.save_object_as_json(data, file_path)
            else:
                general_utils.update_json_file(data, file_path)

            self._store_file_in_file_store(self.filestore_buckets[bucket_name], file_name, file_path)
        except Exception as e:
            print("Failed to store json %s in the file store. Reason: %s" % (str(file_name), str(e)))
            print("Stored %s in %s" % (str(file_name), str(self.temporary_storage_path)))

    def _store_spark_object_into_files(self, spark_object, file_name, bucket_name='objects'):
        try:
            if self.file_store_handler.empty():
                raise SkipToLocalException("No file storage set up in 'config.py'")

            file_path = os.path.join(self.temporary_storage_path, file_name)
            spark_object.save(file_path)

            general_utils.compress_folder_to_zip(file_path)
            file_path += ".zip"
            file_name += ".zip"

            self._store_file_in_file_store(self.filestore_buckets[bucket_name], file_name, file_path)
            os.remove(file_path)
        except SkipToLocalException:
            # storing file in local storage
            file_path = os.path.join(self.local_storage_path, file_name)
            spark_object.save(file_path)
            print("Stored spark model in file %s in local storage %s" % (str(file_name), str(self.local_storage_path)))
        except Exception as e:
            print("Failed to store spark model in file storage. Reason: " + str(e))
            print("Spark model is saved in %s storage path instead." % str(self.temporary_storage_path))
            self.temporary_storage_files.add(str(file_name))

    ##########################################################################
    #                                                                        #
    #                          Helper Functions                              #
    #                                                                        #
    ##########################################################################

    def set_output_folders(self):
        """
        This function set output folders. if file storage is exists, temporary output is created
        else output folder is created.
        """
        if not os.path.exists(self.local_storage_path):
            os.makedirs(self.local_storage_path)
        if not self.file_store_handler.empty() and not os.path.exists(self.temporary_storage_path):
            os.makedirs(self.temporary_storage_path)

    def clean_output_folders(self):
        """
        This function set output folders. if file storage is exists, temporary output is created
        else output folder is created.
        """
        try:
            if os.path.exists(self.local_storage_path) and not os.listdir(self.local_storage_path):
                os.rmdir(self.local_storage_path)
            if os.path.exists(self.temporary_storage_path) and not os.listdir(self.temporary_storage_path):
                os.rmdir(self.temporary_storage_path)
        except:
            pass

    def import_object_from_module(self, module, class_name):
        s = "from " + str(module) + " import " + class_name
        try:
            return s
        except Exception as e:
            print(e)

    def update_config(self, config_model=None, config_data=None):
        if config_data is not None:
            config_data_settings = config_data.get('data_settings', {})
            updated_data_settings = general_utils.recursive_dict_update(config_data_settings,
                                                                        self.config.get('data_settings', {}))
            self.config['data_settings'] = updated_data_settings

        if config_model is not None:
            config_model_model_settings = config_model.get('model_settings', {})
            updated_model_settings = general_utils.recursive_dict_update(config_model_model_settings,
                                                                         self.config.get('model_settings', {}))
            self.config['model_settings'] = updated_model_settings

    def validate_config(self):
        pipeline = self.job_settings.get('pipeline', None)
        if pipeline:
            asset_name = self.identity['asset_name']
            schema_file = "assets/{}/configs/{}_schema.json".format(asset_name, pipeline)
        else:
            raise ConfigKeyError('missing required field - "pipeline" in `task_settings`')
        if os.path.exists(schema_file):
            with open(schema_file, 'r') as f:
                schema = json.loads(f.read())
            try:
                jsonschema.validate(instance=self.config, schema=schema)
            except jsonschema.exceptions.ValidationError as ve:
                raise ConfigError('Error validating your config: {' + str(ve.message) + '}\n')

    def temporary_files_message(self):
        if not self.file_store_handler.empty() and len(self.temporary_storage_files) > 0:
            print("The Following files:", sep=" ")
            print(*self.temporary_storage_files, sep=", ")
            print("failed to saved on file storage and stored in %s instead." % str(self.temporary_storage_path))

    def log_to_file(self, run_id):
        logger_dir = self.local_storage_path if self.file_store_handler.empty() else self.temporary_storage_path
        logger_file_path = os.path.join(logger_dir, self.get_logger_filename(run_id))

        # create log handler
        log = logging.getLogger(self.job_id)
        log.setLevel(logging.DEBUG)
        handler = logging.FileHandler(logger_file_path)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(handler)
        sys.stdout = Logger(log, logging.DEBUG, self.job_id)

    def pipeline_start_print(self, run_id=None):
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.last_time = self.start_time
        print('-------------------------------------- Starting analysis ----------------------------------------------')
        if run_id is None:
            print('Analysis `{}` began running at {}'.format(self.identity['asset_name'], self.start_time))
        else:
            print('Analysis `{}` began running at `{}`, with Run Id: `{}`'
                  .format(self.identity['asset_name'], self.start_time, run_id))
        print('-------------------------------------------------------------------------------------------------------')

    def pipeline_end_print(self, run_id=None):
        time_now = time.strftime('%Y-%m-%d %H:%M:%S')
        start_time_delta = datetime.datetime.strptime(
            time_now, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')

        self.last_time = time_now
        print('------------------------------------ Analysis Summary Print -------------------------------------------')
        print('Job id: {}'.format(self.job_id))
        print('Asset Name: {}'.format(self.identity['asset_name']))
        print('Run id: {}'.format(run_id))
        # print('Local Config path (in case `run.py` was used): ', self.config_path)
        images = self.output_manager.get_images_files()
        if images:
            print('Images saved:  ')
            for img_name in images:
                print(' >> ', img_name)
        print('Current date and time: {}'.format(time_now))
        print('Analysis started running at: {}'.format(self.start_time))
        print('Elapsed Time from the beginning: {}'.format(str(start_time_delta)))
        print('-------------------------------------------------------------------------------------------------------')
        print('Job has completed successfully.')
