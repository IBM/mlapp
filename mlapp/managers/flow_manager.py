import time
import datetime as dt
import uuid

from mlapp.managers.job_manager import JobManager
from mlapp.utils.exceptions.base_exceptions import ConfigKeyError, ConfigError, IoManagerException

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


class FlowManager(object):

    def __init__(self, job_id, config, **kwargs):
        """
        Class FlowManager -
        :param flow_job_id: unique job id
        :param config: json containing the model settings, data settings, job settings
        for each model in the flow, and another model config named "flow_config" and
        see documentation for example config
        :param handlers: can be initiated from the app
        :param kwargs: flow_id as a minimum, and any other input from predecessor steps as followings:
        {'input_from_predecessor':{'key':value,'key':value}}
        """

        if (config is None) or (config == '') or (config == {}):
            raise ConfigError("ModelFlow INIT ERROR: received an empty config")

        self.config = config

        self.has_flow_summary = True
        self.flow_config = self.config.get('flow_config', {})
        if self.flow_config == {}:
            self.has_flow_summary = False

        self.job_id = job_id  # this used to be the job_id

        self.flow_id = kwargs.get('flow_id', str(uuid.uuid4()))

        self.pipelines_configs = self.config.get('pipelines_configs', [])
        self.flow_name = config.get('job_settings', {}).get("flow_name", '')

        # If kwargs have an input from outside it should be stored in kwargs as followings:
        # {'input_from_predecessor':{'key':value,'key':value}}
        self.flow_data = {'jobs_outputs': kwargs.get('input_from_predecessor', [None])}
        self.jobs_results = {'models_results': []}

    def run(self):
        '''
        Runs the flow, passes the input from previous pipelines to the next ones.
        Each pipeline has its own job manager, its own run id , and is returning a dictionary output.
        The config can have instruction regarding input from predecessor keys and output keys to pass to the next pipeline.

        :return: status_code, run_ids - list of all pipelines run ids, pipeline_outputs - output from all pipelines
        '''
        if self.pipelines_configs is None:
            raise ConfigKeyError("'pipelines_configs' property is missing in your config file.'")
        print(">>>>>> Running flow" + self.flow_name + "...")
        run_ids = []
        pipelines_metadata = []
        for i in range(len(self.pipelines_configs)):
            job_config = self.pipelines_configs[i]
            job_flow_setting = job_config.get('flow_settings', {})

            start_time = time.strftime(TIME_FORMAT)
            asset_name = job_config['job_settings'].get('asset_name', i)
            asset_label = job_config['job_settings'].get('asset_label', '')
            print(">>>>>> Running model: {}...".format(asset_name))

            run_ids.append(str(uuid.uuid4()))
            pipelines_metadata.append({
                'run_id': run_ids[i],
                'asset_name': asset_name,
                'asset_label': asset_label
            })
            cur_job_manager = JobManager(self.job_id, job_config, **{'run_id': run_ids[i]})

            kwargs = {'input_from_predecessor': {}}

            if ('input_from_predecessor' in job_flow_setting) and (
                    len(job_flow_setting['input_from_predecessor']) > 0):
                for k in job_flow_setting['input_from_predecessor']:
                    try:
                        if i == len(self.flow_data['jobs_outputs'])-1:
                            kwargs['input_from_predecessor'][k] = self.flow_data['jobs_outputs'][i][k]
                        else:
                            kwargs['input_from_predecessor'][k] = self.flow_data['jobs_outputs'][i-1][k]
                    except KeyError as e:
                        print(e)
                        raise IoManagerException(
                            "Flow job #{}: Error: cannot find input key: {} in the "
                            "results dictionary from your model:{} due to: {}".format(i, k, asset_name, str(e)))

            cur_results = cur_job_manager.run_pipeline(**kwargs['input_from_predecessor'])
            self.jobs_results['models_results'].append(cur_results)
            return_dict = {}
            if 'return_value' in job_flow_setting:
                for output_key in job_flow_setting['return_value']:
                    try:
                        return_dict[output_key] = cur_results.search_key_value(output_key)
                    except:
                        raise IoManagerException(
                            'return value {} from asset: {} is not existing '.format(output_key, asset_name))
            self.flow_data['jobs_outputs'].append(return_dict)

            end_time = dt.datetime.strptime(
                time.strftime(TIME_FORMAT), TIME_FORMAT) - dt.datetime.strptime(start_time, TIME_FORMAT)
            print(">>>>>> It took me, {}.".format(end_time))

        # insert the results data into the data_settings config:
        # self.flow_config['data_settings']['flow_data'] = self.flow_data
        # self.flow_config['job_settings']['flow_models_ids'] = model_ids
        flow_run_id = str(uuid.uuid4())
        flow_job_manager = JobManager(self.job_id, self.flow_config, **{
            'run_id': flow_run_id,
            'has_flow_summary': self.has_flow_summary
        })
        if self.has_flow_summary:
            args = [self.flow_data['jobs_outputs'][1:], run_ids]
            flow_output = flow_job_manager.run_pipeline(*args)
            self.jobs_results['models_results'].append(flow_output)
            run_ids.append(flow_run_id)

        if len(run_ids) > 1:
            flow_job_manager.store_flow_metadata(self.flow_id, **{'pipelines_metadata': pipelines_metadata})

        print(">>>>>> Finished running flow.")

        # todo: make sure to return proper value or just return
        return 100, run_ids, self.flow_data['jobs_outputs'][1:]
