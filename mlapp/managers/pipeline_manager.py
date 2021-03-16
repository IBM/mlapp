import time
import datetime as dt
import importlib.util
import sys
import os
from mlapp.config import settings
from mlapp.utils.exceptions.base_exceptions import PipelineManagerException, FrameworkException
from mlapp.managers.io_manager import IOManager

AVAILABLE_STAGES = {}
BASE_CLASS_NAME = ''
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
MANAGER_TYPES = {
    'data_manager': 'DataManager',
    'model_manager': 'ModelManager'
}


class PipelineManager(object):
    def __init__(self, run_id, pipeline_input, _input: IOManager, _output: IOManager, config, *args, **kwargs):
        """
        :param pipeline_input: the pipeline name string or list of strings
        :param _input: IOmanager instance with input to the pipeline
        :param _output: IOmanager instance to store the outputs of the pipelines to be saved externally
        :param config: config string of the pipeline
        :param args:
        :param kwargs:
        """
        for asset_name in AVAILABLE_STAGES:
            if asset_name != BASE_CLASS_NAME:
                AVAILABLE_STAGES[asset_name] = {}

        self.pipeline_name = ''

        # pipeline can be either list of stages or string of a default pipeline
        if isinstance(pipeline_input, list):
            self.stages = pipeline_input
        if isinstance(pipeline_input, str):
            self.pipeline_name = " '" + pipeline_input + "'"
            self.stages = settings.get('pipelines', {}).get(pipeline_input, [])

        self.config = config
        self.run_id = run_id
        self.input_manager = _input
        self.output_manager = _output
        self.asset_name = self.config.get('job_settings', {}).get('asset_name', '')

        self.data_manager_instance = self.create_manager_instance('data')
        self.model_manager_instance = self.create_manager_instance('model')

        # first inputs
        self.state = dict.fromkeys(self.stages, {})

    def create_manager_instance(self, manager_type):
        """
        Creates manager instance which class is defined in the asset. For example: model_manager or data_manader
        :param manager_type: the type : e.g. "data", "model"
        :return: instance of the manager
        """

        manager_file_name = self.asset_name + '_' + manager_type + '_manager'
        manager_module = 'assets.' + self.asset_name + '.' + manager_file_name
        manager_module_path = os.path.join('assets', self.asset_name, f'{manager_file_name }.py')
        manager_class_name = ''.join(x.capitalize() or '_' for x in manager_file_name.split('_'))  # CamelCase

        try:
            spec = importlib.util.spec_from_file_location(manager_module, manager_module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            manager_class = getattr(module, manager_class_name)
            return manager_class(self.config.copy(), self.input_manager, self.output_manager, self.run_id)

        except Exception as e:
            print("Couldn't import class of " + manager_type + " manager for model: " + self.asset_name)
            print(">>> Please verify your " + manager_type +
                  " manager file/directory/class names are in the following convention:")
            print(">>> Directory: {{asset_name}}")
            print(">>> File: {{asset_name}}_" + manager_type + "_manager.py")
            print(">>> Class: {{asset_name_capitalized}}" + manager_type.capitalize() + "Manager")
            print(">>> Expected Directory name: " + self.asset_name)
            print(">>> Expected Manager file name: " + manager_file_name)
            print(">>> Expected Manager class name: " + manager_class_name)
            raise FrameworkException(str(e))

    def extract_stage(self, stage_name):
        """
        Gets the pipeline stage dictioanry containing the function and the manager class it resides in.
        :param stage_name: the name of the stage (e.g. "load_data")
        :return:  pipeline stage dictionary
        """
        asset_name = ''.join(x.capitalize() or '_' for x in self.asset_name.split('_'))  # CamelCase
        if asset_name not in AVAILABLE_STAGES:
            raise PipelineManagerException(
                "Missing decoration for your pipeline functions! Add '@pipeline' decorator above functions"
                " you want to use in your asset '{}'s Data Manager and Model Manager.".format(asset_name))

        if stage_name not in AVAILABLE_STAGES[asset_name]:
            # exists in one if the base classes
            if stage_name in AVAILABLE_STAGES[BASE_CLASS_NAME]:
                return AVAILABLE_STAGES[BASE_CLASS_NAME][stage_name]

            raise PipelineManagerException(
                "Function '{}' was not found in your asset! Add '@pipeline' decorator above your '{}' "
                "function if you want to use it in your pipeline.".format(stage_name, stage_name))

        return AVAILABLE_STAGES[asset_name][stage_name]

    def extract_manager_instance(self, manager_type):
        """
        Gets the instance of the manager - model_manager instance or data_manager instance
        :param manager_type: string "data_manager" or "model_manager"
        :return: model_manager instance or data_manager instance
        """
        if manager_type == 'data_manager':
            return self.data_manager_instance
        else:
            return self.model_manager_instance

    def run(self, *arguments):
        """
        Runs through the pipeline stages and passes relevant values between them
        :param arguments: input for the first stage in the pipeline, will be passed with *args
        :return: IOmanager of all the outputs to be stored
        """
        print(">>>>>> Running pipeline" + self.pipeline_name + "...")
        prev_stage_name = ''
        for stage_name in self.stages:
            start_time = time.strftime(TIME_FORMAT)
            print(">>>>>> Running stage: {}...".format(stage_name))

            stage = self.extract_stage(stage_name)
            if prev_stage_name:
                args = self.state[prev_stage_name]
            else:
                args = arguments

            # execute stage
            self.state[stage_name] = stage['function'](self.extract_manager_instance(stage['manager']), *args)

            if not isinstance(self.state[stage_name], tuple):
                self.state[stage_name] = (self.state[stage_name],)

            prev_stage_name = stage_name

            end_time = dt.datetime.strptime(
                time.strftime(TIME_FORMAT), TIME_FORMAT) - dt.datetime.strptime(start_time, TIME_FORMAT)
            print(">>>>>> It took me, {}.".format(end_time))

        print(">>>>>> Finished running pipeline.")
        return self.output_manager


class pipeline:
    def __init__(self, fn):
        self.fn = fn

    def __set_name__(self, owner, name):
        asset_name = owner.__name__

        manager_type = None
        for manager_type_key in MANAGER_TYPES:
            if MANAGER_TYPES[manager_type_key] in asset_name:
                manager_type = manager_type_key
        if manager_type is None:
            raise Exception("Wrong class name or placement of decorator! ('{}')".format(asset_name))

        asset_name = asset_name.replace('DataManager', '').replace('ModelManager', '')

        if asset_name not in AVAILABLE_STAGES:
            AVAILABLE_STAGES[asset_name] = {}

        if name in AVAILABLE_STAGES[asset_name]:
            raise Exception("Duplicate stage name '{}' for pipelines found in asset '{}'"
                            .format(asset_name, name))

        AVAILABLE_STAGES[asset_name][name] = {
            'function': self.fn,
            'manager': manager_type
        }

        return self.fn
