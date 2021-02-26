import time
from kafka import KafkaConsumer, TopicPartition
from kafka import KafkaProducer
from mlapp.handlers.message_queues.message_queue_interface import MessageQueueInterface


class KafkaHandler(MessageQueueInterface):
    def __init__(self, settings):
        """
        Initializes the ￿KafkaHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(KafkaHandler, self).__init__()

        # init
        self.connection_string = settings.get('hostname') + ':' + str(settings.get('port'))

        self.connection_timeout = settings.get('connection_timeout', 15)

        self.username = settings.get('username')
        self.password = settings.get('password')

        self.security_protocol = "PLAINTEXT"
        self.sasl_mechanism = None

        if self.username != "" and self.password != "":
            self.sasl_mechanism = "PLAIN"
            self.security_protocol = "SASL_PLAINTEXT"

    def send_message(self, queue_name, body):
        """
        Sends message to the queue
        :param queue_name: name of the topic/queue to send the message to
        :param body: message as string or bytes
        """
        producer = KafkaProducer(
            bootstrap_servers=[self.connection_string],
            sasl_plain_username=self.username,
            sasl_plain_password=self.password,
            sasl_mechanism=self.sasl_mechanism,
            security_protocol=self.security_protocol
        )
        b = bytearray()
        b.extend(map(ord, body))
        producer.send(queue_name, value=b, partition=0)
        producer.flush()

    def listen_to_queues(self, queue_names, callback):
        """
        Listen to queues/topics
        :param queue_names: list of queue/topic names to listen to
        :param callback: function to call upon receiving a message
        """
        consumer = KafkaConsumer(
            *queue_names,
            # group_id="my_group",
            bootstrap_servers=[self.connection_string],
            sasl_plain_username=self.username,
            sasl_plain_password=self.password,
            sasl_mechanism=self.sasl_mechanism,
            security_protocol=self.security_protocol
        )

        self.listening = False
        while True:
            try:
                message = consumer.poll()

                for tp, messages in message.items():
                    for message in messages:
                        # message value and key are raw bytes -- decode if necessary!
                        # e.g., for unicode: `message.value.decode('utf-8')`
                        print("%s:%d:%d: key=%s value=%s" % (tp.topic, tp.partition,
                                                             message.offset, message.key,
                                                             message.value))

                        callback(message.value)
                        self.listening = False

                if self.listening == False:
                    self.listening = True
                    print('[*] Waiting for messages in ' + str(queue_names) + '. To exit press CTRL+C')

            except Exception as e:
                print(e)
                time.sleep(1)
                continue

