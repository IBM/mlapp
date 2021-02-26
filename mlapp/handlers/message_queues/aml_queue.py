import json
import os
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'
from mlapp.env_loader import EMPTY_STRING
from mlapp.handlers.message_queues.message_queue_interface import MessageQueueInterface
from mlapp.integrations.aml.utils.pipeline import run_pipeline_endpoint
from azureml.core import Run
from azureml.exceptions import RunEnvironmentException
from mlapp.utils.exceptions.framework_exceptions import SkipServiceException


class AMLQueue(MessageQueueInterface):
    def __init__(self, settings):
        """
        Initializes the ￿AMLQueue with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(AMLQueue, self).__init__()

        # getting azureml workspace
        try:
            self.global_run = Run.get_context(allow_offline=False)
        except RunEnvironmentException as e:
            raise SkipServiceException('Skip AMLQueue handler')
        self.ws = self.global_run.experiment.workspace
        self.experiment_name = settings.get('experiment_name', self.global_run.experiment.name)
        if self.experiment_name == EMPTY_STRING:
            self.experiment_name = self.global_run.experiment.name

    def send_message(self, queue_name, body):
        """
        Sends message to the queue
        :param queue_name: name of the topic/queue to send the message to
        :param body: message as string or bytes
        """
        # submitting a new pipeline endpoint run
        run_pipeline_endpoint(
            workspace=self.ws,
            pipeline_endpoint_id=queue_name,
            experiment_name=self.experiment_name,
            config_str=body,
            pipeline_version=None
        )

    def listen_to_queues(self, queue_names, callback):
        """
        Listen to queues/topics - not relevant in AML Queue
        :param queue_names: list of queue/topic names to listen to
        :param callback: function to call upon receiving a message
        """
        raise NotImplemented()
