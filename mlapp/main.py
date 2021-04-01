import multiprocessing
from mlapp.env_loader import EnvironmentLoader
from mlapp.utils.general import read_json_file
import importlib
import uuid
import json
from mlapp.handlers.wrappers.database_wrapper import database_instance
from mlapp.handlers.wrappers.file_storage_wrapper import file_storage_instance
from mlapp.handlers.wrappers.message_queue_wrapper import message_queue_instance
from mlapp.handlers.wrappers.spark_wrapper import spark_instance
from mlapp.managers.flow_manager import FlowManager
import traceback
from ast import literal_eval
from mlapp.config import settings, environment_services
import os


class MLApp(object):
    MLAPP_SERVICE_TYPE = '_MLAPP_SERVICE_TYPE'

    def __init__(self, inner_settings=None):
        """
        Constructor for the MLApp Class.
        This class, when becomes instantiated is the main endpoint for the ML App Library.
        To this constructor you pass in your custom settings, which sets up your customized configuration of ML App,
        and Environment that you defined with the ML App CLI.
        After the instantiation you can use the instance to run:
         - Flows (single-process and multi-process)
         - Applications/Workers (using the queue listener)
         - Send configurations to an outside queue
        """
        if inner_settings is None:
            inner_settings = {}

        for key in inner_settings:
            if isinstance(inner_settings[key], dict):
                if key not in settings:
                    settings[key] = {}
                settings[key].update(inner_settings[key])
            elif isinstance(inner_settings[key], list):
                if key not in settings:
                    settings[key] = []
                settings[key] += inner_settings[key]
            else:
                settings[key] = inner_settings[key]

        # init environment services
        env = EnvironmentLoader.load(filename=inner_settings.get('env_file_path', ''))
        env_services_dict = {
            k.replace(self.MLAPP_SERVICE_TYPE, ''): os.environ[k].lower()
            for k in os.environ if k.endswith(self.MLAPP_SERVICE_TYPE)
        }
        settings['services'] = EnvironmentLoader.create_services(env, env_services_dict, environment_services)
        for wrapper_instance in [file_storage_instance, database_instance, message_queue_instance, spark_instance]:
            wrapper_instance.init()

    # ======== TASK RUN  ==========
    def _on_callback(self, message_body):
        """
        This is the function that executes the configuration sent to the application/worker.
        :param message_body: the configuration in bytes/string format.
        :return: None
        """
        job_id = 'None'
        try:
            message_body = message_body.decode("utf-8")
        except AttributeError as attrError:
            pass    # message body is string and not bytes

        print("Hello, The following task is consumed:   " + str(message_body))
        try:
            message_config = json.loads(literal_eval(message_body))
        except Exception as first_error:
            try:
                message_config = json.loads(message_body)
            except Exception as second_error:
                print("Error response: " + str('Message not in JSON format'))
                traceback.print_exc()
                self._send_error_response_to_mq(job_id, '-1', str('Message not in JSON format'))
                return

        print(message_config)
        try:
            job_id = str(message_config.get('job_id', str(uuid.uuid4())))
            results = self._run_flow(job_id, message_config)
            self._send_ok_response_to_mq(
                job_id, results.get('status_code', -1), 'all went ok', results.get('response', {}))

        except Exception as error:
            print("Error response: " + str(error))
            traceback.print_exc()
            self._send_error_response_to_mq(job_id, '-1', str(error))
        finally:
            pass

    # =========== TASK HANDLERS ===========
    def _run_flow(self, job_id, config):
        """
        This is the function that executes the Flow of your configuration.
        :param job_id: the job identifier used for monitoring via the MLCP (Machine Learning Control Panel).
        :param config: the configuration as Dictionary.
        :return: Dictionary containing the status and response of the flow run.
        """
        # update job
        database_instance.update_job_running(job_id)
        # call Flow_manager to run the job
        status_code, response, _ = FlowManager(job_id, config).run()

        return {'status_code': status_code, 'response': response}

    # ======== MQ HANDLERS =========
    def _send_ok_response_to_mq(self, job_id, status_code, status_msg, result):
        """
        This function sends response back to the MLCP (Machine Learning Control Panel) via queue if the job succeeded
        :param job_id: the job identifier used for monitoring via the MLCP.
        :param status_code: result status of the flow run.
        :param status_msg: result message of the flow run.
        :param result: response of the flow run - if json serialized returned in the message queue as well.
        :return: None
        """
        response_obj = {
            "job_id": job_id, "status_code": status_code, "status_msg": status_msg
        }
        try:
            # trying to JSON-ify result object
            response_obj['result'] = result
            response_json = json.dumps(response_obj)
        except Exception as error:
            print(error)
            response_obj['result'] = {}
            response_json = json.dumps(response_obj)

        message_queue_instance.send_message(settings['queues']['send_queue_name'], response_json)

    def _send_error_response_to_mq(self, job_id, status_code, status_msg):
        """
        This function sends response back to the MLCP (Machine Learning Control Panel) via queue if the job failed
        :param job_id: the job identifier used for monitoring via the MLCP.
        :param status_code: error status of the flow run.
        :param status_msg: error message of the flow run.
        :return: None
        """
        response_json = json.dumps({"job_id": job_id, "status_code": status_code, "status_msg": status_msg})
        message_queue_instance.send_message(settings['queues']['send_queue_name'], response_json)

    def _dispatch_jobs_to_mq(self, configurations):
        """
        This function sends configurations to the queue to be picked up later by a listening Application/Worker.
        :param configurations: list of configurations to be sent
        :return: None
        """
        for configuration in configurations:
            response_json = json.dumps(configuration)
            message_queue_instance.send_message(settings['queues']['send_queue_name'], json.dumps(response_json))

    # ======== LISTENER  =========
    def run_listener(self):
        """
        This function is an endpoint of the ML App Library to be used in an Application/Worker.
        It sets up a listening queue indefinitely waiting for configuration to process upon receive.
        """
        message_queue_instance.listen_to_queues(settings['queues']['listen_queue_names'], self._on_callback)

    # ======== RUN CONFIG  =========
    def run_flow(self, asset_name, config_path, config_name=None, **kwargs):
        """
        This function is an endpoint of the ML App Library to be used in a local environment.
        It runs a local configuration file in your local computer.
        :param asset_name: name of the asset to be run
        :param config_path: path to configuration file
        :param config_name: in case configuration file is python looks for variable in this name as the configuration
        """
        job_id = str(uuid.uuid4())
        try:
            config = read_json_file(config_path)
        except Exception as err:
            config = self._read_py_file(asset_name, config_path, config_name)
        self._insert_latest_id_in_config(config)
        _, run_ids, outputs = FlowManager(job_id, config, **kwargs).run()
        self._update_latest_model_id(config, run_ids)

    @staticmethod
    def run_flow_from_config(config):
        return FlowManager("deployment", config).run()

    # ======== SEND CONFIG TO MQ  =========
    def run_msg_sender(self, asset_name, config_path, config_name=None):
        """
        This function is an endpoint of the ML App Library to be used in a local environment.
        It sends a local configuration file in your local computer to be run in an outside Application/Worker via
        message queue.
        :param asset_name: name of the asset to be run
        :param config_path: path to configuration file
        :param config_name: in case configuration file is python looks for variable in this name as the configuration
        """
        try:
            message_to_send = read_json_file(config_path)
        except Exception as e:
            message_to_send = self._read_py_file(asset_name, config_path, config_name)

        job_id = str(uuid.uuid4())
        message_to_send['job_id'] = job_id
        message_queue_instance.send_message(settings['queues']['listen_queue_names'][0], json.dumps(message_to_send))
        print("Message Sent (job_id: " + job_id + "): ", asset_name, config_path)

    # ======== RUN CONFIGS MULTIPROCESSING  =========
    def run_configs_multiprocessing(self, instructions):
        """
        This function is an endpoint of the ML App Library.
        It runs multiple configurations in multi-processing.
        :param instructions: list of instruction to send to each process
        """
        jobs = []
        for instruction in instructions:
            p = multiprocessing.Process(target=self._run_config_multiprocess, args=(instruction,))
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

    # ======== HELPER PRIVATE FUNCTIONS  =========
    def _run_config_multiprocess(self, instruction):
        """
        This function is executes instruction of a process when used by `run_configs_multiprocessing`.
        :param instruction: instruction Dictionary containing asset_name, config_path and config_name.
            - asset_name: name of the asset to be run
            - config_path path to configuration file
            - config_name in case configuration file is python looks for variable in this name as the configuration
        """
        try:
            self.run_flow(instruction['asset_name'], instruction['config_path'], instruction.get('config_name'))
        except Exception as err:
            print(err)
            traceback.print_exc()

    @staticmethod
    def _read_py_file(asset_name, config_path, config_name):
        """
        This function fetches a configuration Dictionary stored in a python file.
        :param asset_name: name of the asset to be run
        :param config_path: path to configuration file
        :param config_name: variable in the python file containing the configuration
        :return: Configuration as a Dictionary
        """
        spec = importlib.util.spec_from_file_location(asset_name, config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config.__dict__[config_name]

    @staticmethod
    def _insert_latest_id_in_config(config):
        """
        This is a helper function for using `latest` feature in local environment.
        Updates current configuration to be run with the latest id stored in a local file containing it
        by reference of asset name.
        :param config: current flow run configuration as a Dictionary
        """
        # prepare latest file
        local_path = settings.get('local_storage_path', 'output')
        latest_file_name = settings.get('latest_file_name', 'latest_ids.json')
        latest_ids_path = os.path.join(local_path, latest_file_name)
        try:
            with open(latest_ids_path) as f:
                latest = json.load(f)
        except:
            latest = {}

        # iterate pipelines
        for i, pipeline in enumerate(config.get('pipelines_configs', [])):
            # iterate optional ids
            for id_type in ['model_id', 'data_id', 'reuse_features_id']:
                # check if requested latest
                if pipeline.get('job_settings', {}).get(id_type, None) == 'latest':
                    # get current asset name
                    asset_name = pipeline['job_settings']['asset_name']
                    # check if available id
                    if asset_name in latest:
                        # TODO: add here asset label level
                        # TODO: add here data_id/model_id/reuse_features_id
                        config['pipelines_configs'][i]['job_settings'][id_type] = latest[asset_name]
                    else:
                        # raise exception as not found id
                        raise Exception("Could not find latest `" + id_type + "` for `" + asset_name + "`. \n"
                                        "Please update your config with a valid `" + id_type + "`")

    @staticmethod
    def _update_latest_model_id(config, run_ids):
        """
        This is a helper function for using `latest` feature in local environment.
        Updates local file containing the latest id used for an asset.
        :param config: current flow run configuration as a Dictionary
        :params run_ids: list of mlapp identifiers generated in the current flow run.
        """
        # prepare latest file
        local_path = settings.get('local_storage_path', 'output')
        latest_file_name = settings.get('latest_file_name', 'latest_ids.json')
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        latest_ids_path = os.path.join(local_path, latest_file_name)

        latest = {}
        try:
            with open(latest_ids_path) as f:
                latest = json.load(f)
        except:
            pass

        # iterate over pipelines
        for pipeline, run_id in zip(config['pipelines_configs'], run_ids):
            # check if ran any pipeline where id is being stored
            # TODO: add here asset label level
            # TODO: add here data_id/model_id/reuse_features_id
            if pipeline['job_settings']['pipeline'] in ['train', 'feature_engineering']:
                latest[pipeline['job_settings']['asset_name']] = run_id

            with open(latest_ids_path, 'w') as f:
                json.dump(latest, f)
