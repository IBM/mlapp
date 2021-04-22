import logging
from minio import Minio
from minio.error import S3Error
from mlapp.handlers.file_storages.file_storage_interface import FileStorageInterface


class MinioStorageHandler(FileStorageInterface):
    def __init__(self, settings):
        """
        Initializes the ￿MinioStorageHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(MinioStorageHandler, self).__init__()
        configuration = settings

        if not configuration:
            logging.error('Configuration should be added to the file. Key should be "minio"')

        try:
            self.minioClient = Minio(configuration['endPoint'] + ":" + configuration['port'],
                                     access_key=configuration['accessKey'],
                                     secret_key=configuration['secretKey'],
                                     secure=configuration['secure'],
                                     region=configuration['region'])
            logging.info("File storage: Successful connection")
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
        self.minioClient.fget_object(bucket_name, object_name, file_path)

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
            self.minioClient.make_bucket(bucket_name)
        except S3Error as e:
            if e.code == 'BucketAlreadyOwnedByYou':
                pass
            else:
                raise e
        try:
            # with open(from_file_path, 'rb') as from_file:
            self.minioClient.fput_object(bucket_name, object_name, file_path)
            logging.info('File storage: file %s successfully uploaded', object_name)
        except IOError as e:
            logging.error(e)

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
            objects = self.minioClient.list_objects(bucket_name, prefix=prefix, recursive=True)
            for obj in objects:
                files_names.append(obj.object_name.encode('utf-8'))
        except Exception as e:
            logging.error(e)

        return files_names
