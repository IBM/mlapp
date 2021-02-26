import time
import os
import json
import confluent_kafka
from mlapp.handlers.message_queues.message_queue_interface import MessageQueueInterface


class KafkaKerberosHandler(MessageQueueInterface):
    def __init__(self, settings):
        """
        Initializes the ￿KafkaKerberosHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(KafkaKerberosHandler, self).__init__()

        self.config = {
            'bootstrap.servers': settings.get('hostnames', ''),
            'group.id': settings.get('group_id'),
            'session.timeout.ms': 6000,
            'enable.auto.commit': 'TRUE',
            # Your AD account principle name
            'sasl.kerberos.principal': '{0}@{1}'.format(settings.get('username', ''), settings.get('domain_name', '')),
            # Path to your keytab file
            'sasl.kerberos.keytab': settings.get('keytab'),
            'sasl.kerberos.service.name': 'kafka',
            'security.protocol': 'SASL_SSL',
            'sasl.mechanisms': 'GSSAPI',
            # Certificate used for creating JKS files
            'ssl.ca.location': settings.get('ca_location')
        }

    def send_message(self, queue_name, body):
        """
        Sends message to the queue
        :param queue_name: name of the topic/queue to send the message to
        :param body: message as string or bytes
        """
        producer = confluent_kafka.Producer(self.config)
        raw_bytes = json.dumps(body).encode('utf-8')
        producer.produce(queue_name, raw_bytes)
        producer.flush()

    def listen_to_queues(self, queue_names, callback):
        """
        Listen to queues/topics
        :param queue_names: list of queue/topic names to listen to
        :param callback: function to call upon receiving a message
        """
        consumer = confluent_kafka.Consumer(self.config)
        consumer.subscribe(queue_names)
        print('[*] Waiting for messages in ' + str(queue_names) + '. To exit press CTRL+C')

        while True:
            msg = consumer.poll()

            if msg is None:
                continue
            if msg.error():
                print("Consumer error: {}".format(msg.error()))
                continue

            callback(msg.value().decode('utf-8'))