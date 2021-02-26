import logging
import ibm_boto3
from ibm_botocore.client import Config, ClientError
from mlapp.handlers.file_storages.file_storage_interface import FileStorageInterface


class IBMBoto3StorageHandler(FileStorageInterface):
    def __init__(self, settings):
        """
        Initializes the ￿BotoStorageHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(IBMBoto3StorageHandler, self).__init__()
        configuration = settings

        if not configuration:
            logging.error('Configuration should be added to the file. Key should be "boto"')

        try:
            # Create client
            self.botoClient = ibm_boto3.client("s3",
                                               ibm_api_key_id=configuration["api_key_id"],
                                               ibm_service_instance_id=configuration['service_crn'],
                                               config=Config(signature_version="oauth"),
                                               endpoint_url=configuration['endpoint'])

        except KeyError as e:
            logging.error("Missing parameter in file storage config %s" % str(e))

    def download_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        Downloads file from file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        self.botoClient.download_file(bucket_name, object_name, file_path)

    def upload_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        Uploads file to file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        try:
            self.botoClient.create_bucket(Bucket=bucket_name)
        except Exception as e:
            pass
        try:
            self.botoClient.upload_file(file_path, bucket_name, object_name)
            logging.info('File storage: file %s successfully uploaded', object_name)
        except ClientError as e:
            logging.error(e)

    def list_files(self, bucket_name, prefix="", *args, **kwargs):
        """
        Lists files in file storage
        :param bucket_name: name of the bucket/container
        :param prefix: prefix string to search by
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        try:
            files_names = []
            response = self.botoClient.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            for file in response['Contents']:
                files_names.append(file['Key'])
            return files_names
        except Exception as e:
            logging.error(e)
