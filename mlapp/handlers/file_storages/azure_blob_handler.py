import logging
from mlapp.handlers.file_storages.file_storage_interface import FileStorageInterface
from azure.storage.blob import BlockBlobService


class AzureBlobHandler(FileStorageInterface):
    def __init__(self, settings):
        """
        Initializes the ￿AzureBlobHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(AzureBlobHandler, self).__init__()
        self.connections_parameters = settings
        if self.connections_parameters == {}:
            logging.error('Missing connection parameters!')
        self.block_blob_service = BlockBlobService(account_name=self.connections_parameters['accountName'],
                                                   account_key=self.connections_parameters['accountKey'])

    def download_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        Downloads file from file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        self.block_blob_service.get_blob_to_path(bucket_name, object_name, file_path)

    def stream_file(self, bucket_name, object_name, *args, **kwargs):
        """
        Streams file from file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        return self.block_blob_service.get_blob_to_text(bucket_name, object_name)

    def upload_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        Uploads file to file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        # create container (bucket) if it doesn't exist:
        self.block_blob_service.create_container(bucket_name)
        # upload the file:
        self.block_blob_service.create_blob_from_path(bucket_name, object_name, file_path)

    def list_files(self, bucket_name, prefix="", *args, **kwargs):
        """
        Lists files in file storage
        :param bucket_name: name of the bucket/container
        :param prefix: prefix string to search by
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        files_names = []

        try:
            blobs = self.block_blob_service.list_blob_names(bucket_name, prefix=prefix)
            for blob in blobs:
                files_names.append(blob)
        except Exception as e:
            logging.error(e)

        return files_names
