from mlapp.handlers.message_queues.message_queue_interface import MessageQueueInterface
from azure.servicebus import ServiceBusClient, ReceiveSettleMode
from azure.servicebus import Message
from azure.servicebus.common.errors import ServiceBusError
import time


class AzureServicesBusHandler(MessageQueueInterface):
    def __init__(self, settings):
        """
        Initializes the ￿AzureServicesBusHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(AzureServicesBusHandler, self).__init__()

        # init
        self.client = ServiceBusClient(
            service_namespace=settings.get('hostname'),
            shared_access_key_name=settings.get('shared_access_key_name'),
            shared_access_key_value=settings.get('shared_access_key'),
            debug=False)

        self.connection_timeout = settings.get('connection_timeout', 10)

    def send_message(self, queue_name, body):
        """
        Sends message to the queue
        :param queue_name: name of the topic/queue to send the message to
        :param body: message as string or bytes
        """
        while True:
            try:
                queue_client = self.client.get_queue(queue_name)
                with queue_client.get_sender() as sender:
                    message = Message(body)
                    sender.send(message)
                break
            except Exception as e:
                print(e)

    def listen_to_queues(self, queue_names, callback):
        print('[*] Waiting for messages in ' + str(queue_names[0]) + '. To exit press CTRL+C')
        while True:
            try:
                queue_client = self.client.get_queue(queue_names[0])
                with queue_client.get_receiver(mode=ReceiveSettleMode.PeekLock) as receiver:
                    batch = receiver.fetch_next(max_batch_size=1, timeout=self.connection_timeout)
                    while batch:
                        for message in batch:
                            message.complete()
                            callback(str(message))
                        batch = receiver.fetch_next(max_batch_size=1, timeout=self.connection_timeout)
            except ServiceBusError as e:
                print(str(e))
            finally:
                time.sleep(1)
