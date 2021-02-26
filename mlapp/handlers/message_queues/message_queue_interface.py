from abc import ABCMeta, abstractmethod


class MessageQueueInterface:
    __metaclass__ = ABCMeta

    @abstractmethod
    def send_message(self, queue_name, body):
        """
        Sends message to the queue
        :param queue_name: name of the topic/queue to send the message to
        :param body: message as string or bytes
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def listen_to_queues(self, queue_names, callback):
        """
        Listen to queues/topics
        :param queue_names: list of queue/topic names to listen to
        :param callback: function to call upon receiving a message
        :return: None
        """
        raise NotImplementedError()

