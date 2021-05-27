import functools
import time
import pika.exceptions
import pika
import ssl
from mlapp.handlers.message_queues.message_queue_interface import MessageQueueInterface


class RabbitMQHandler(MessageQueueInterface):
    def __init__(self, settings):
        """
        Initializes the ￿RabbitMQHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(RabbitMQHandler, self).__init__()

        connection_params = {
            'host': settings.get('hostname'),
            'port': settings.get('port')
        }

        if settings.get('cert_path', False):
            context = ssl.create_default_context(
                cafile=settings['cert_path']
            )
            ssl_options = pika.SSLOptions(context, settings.get('hostname'))
            credentials = pika.PlainCredentials(settings.get('username'), settings.get('password'))
            self.params = pika.ConnectionParameters(
                **connection_params, credentials=credentials, ssl_options=ssl_options)
        else:
            self.params = pika.ConnectionParameters(**connection_params)

        self.connection_timeout = settings.get('connection_timeout', 15)

    def send_message(self, queue_name, body):
        """
        Sends message to the queue
        :param queue_name: name of the topic/queue to send the message to
        :param body: message as string or bytes
        """
        while True:
            connection = None
            channel = None
            try:
                # opening connection
                connection = pika.BlockingConnection(parameters=self.params)
                channel = connection.channel()

                # connect to queue
                channel.queue_declare(queue=queue_name, durable=True)
                channel.confirm_delivery()

                # sending message
                channel.basic_publish(exchange='', routing_key=queue_name, body=body)
                break
            except pika.exceptions.AMQPChannelError as err:
                print("Caught a channel error: {}, stopping...".format(err))
                break
                # recover on all other connection errors
            except pika.exceptions.AMQPConnectionError:
                print("Connection was closed, retrying...")
                time.sleep(1)
                continue
            except Exception as e:
                print(e)
                time.sleep(1)
                continue
            finally:
                # closing connection
                if connection and connection.is_open and channel:
                    channel.close()
                    connection.close()

    def listen_to_queues(self, queue_names, callback):
        """
        Listen to queues/topics
        :param queue_names: list of queue/topic names to listen to
        :param callback: function to call upon receiving a message
        """
        while True:
            try:
                # connection to rabbitMQ
                conn = pika.BlockingConnection(parameters=self.params)
                chan = conn.channel()

                # preparing listen to queues
                on_message_callback = functools.partial(self._on_message, args=(conn, callback))

                for queue in queue_names:
                    chan.queue_declare(queue=str(queue), durable=True)
                    chan.basic_qos(prefetch_count=1)
                    chan.basic_consume(str(queue), on_message_callback)
                    print('[*] Waiting for messages in ' + str(queue) + '. To exit press CTRL+C')

                # listening to queues
                try:
                    chan.start_consuming()
                except KeyboardInterrupt:
                    chan.stop_consuming()
                    conn.close()
                    break

            except pika.exceptions.AMQPChannelError as err:
                print("Caught a channel error: {}, stopping...".format(err))
                break
                # recover on all other connection errors
            except pika.exceptions.AMQPConnectionError:
                print("Connection was closed, retrying...")
                time.sleep(1)
                continue
            except Exception as e:
                print(e)
                time.sleep(1)
                continue

    @staticmethod
    def _ack_message(body, channel, delivery_tag):
        """Note that `channel` must be the same pika channel instance via which
        the message being ACKed was retrieved (AMQP protocol constraint).
        """
        if channel.is_open:
            channel.basic_ack(delivery_tag)
        else:
            # Channel is already closed, so we can't ACK this message;
            # log and/or do something that makes sense for your app in this case.
            print("Channel was closed during the process of this task. Message can't be acknowledged.")

    @staticmethod
    def _on_message(channel, method_frame, header_frame, body, args):
        (connection, callback) = args
        # auto ack is set to true so no need to ack message
        # self._ack_message(body, channel, method_frame.delivery_tag)

        if channel.is_open:
            channel.basic_ack(method_frame.delivery_tag)
        else:
            # Channel is already closed, so we can't ACK this message;
            # log and/or do something that makes sense for your app in this case.
            print("Channel was closed during the process of this task. Message can't be acknowledged.")

        connection.close()
        callback(body)
