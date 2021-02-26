from abc import ABCMeta, abstractmethod


class FileStorageInterface:
    __metaclass__ = ABCMeta

    @abstractmethod
    def download_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        Downloads file from file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def stream_file(self, bucket_name, object_name, *args, **kwargs):
        """
        Streams file from file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        :return: file stream
        """
        raise NotImplementedError()

    @abstractmethod
    def upload_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        Uploads file to file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        :return: None
        """
        raise NotImplementedError()

    @abstractmethod
    def list_files(self, bucket_name, prefix="", *args, **kwargs):
        """
        Lists files in file storage
        :param bucket_name: name of the bucket/container
        :param prefix: prefix string to search by
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        :return: file names list
        """
        raise NotImplementedError()


