from mlapp.handlers.wrappers.wrapper_interface import WrapperInterface


class MessageQueueWrapper(WrapperInterface):
    def init(self):
        """
        Initializes the Wrapper for all handlers of `message_queue` type.
        """
        super(MessageQueueWrapper, self).init('message_queue')

    def send_message(self, queue_name, body):
        """
        Sends message to the queue
        :param queue_name: name of the topic/queue to send the message to
        :param body: message as string or bytes
        """
        if len(self._main_handlers) == 0:
            raise Exception("Called listen to queue without any queue service setup in ML App!")

        for handler_name in self._main_handlers:
            self._handlers[handler_name].send_message(queue_name, body)

    def listen_to_queues(self, queue_names, callback):
        """
        Listen to queues/topics
        :param queue_names: list of queue/topic names to listen to
        :param callback: function to call upon receiving a message
        """
        if len(self._main_handlers) == 0:
            raise Exception("Called listen to queue without any queue service setup in ML App!")

        for handler_name in self._main_handlers:
            self._handlers[handler_name].listen_to_queues(queue_names, callback)


message_queue_instance = MessageQueueWrapper()


