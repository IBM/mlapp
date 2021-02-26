from mlapp.handlers.wrappers.wrapper_interface import WrapperInterface


class FileStorageWrapper(WrapperInterface):
    def init(self):
        """
        Initializes the Wrapper for all handlers of `file_storage` type.
        """
        super(FileStorageWrapper, self).init('file_storage')

    def download_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        Downloads file from file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        for handler_name in self._main_handlers:
            self._handlers[handler_name].download_file(bucket_name, object_name, file_path, *args, **kwargs)

    def upload_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        Uploads file to file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        for handler_name in self._main_handlers:
            self._handlers[handler_name].upload_file(bucket_name, object_name, file_path, *args, **kwargs)
    
    def stream_file(self, bucket_name, object_name, *args, **kwargs):
        """
        Streams file from file storage
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        for handler_name in self._main_handlers:
            if hasattr(self._handlers[handler_name], 'stream_file'):
                return self._handlers[handler_name].stream_file(bucket_name, object_name, *args, **kwargs)

    def list_files(self, bucket_name, prefix="", *args, **kwargs):
        """
        Lists files in file storage
        :param bucket_name: name of the bucket/container
        :param prefix: prefix string to search by
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        files_names = []
        for handler_name in self._main_handlers:
            files_names.extend(self._handlers[handler_name].list_files(bucket_name, prefix=prefix, *args, **kwargs))
        return files_names

    def postprocessing(self):
        """
        This method is used to run postprocessing for different handlers.
        :return: None
        """
        for key in self._handlers:
            if hasattr(file_storage_instance._handlers[key], 'postprocess'):
                file_storage_instance._handlers[key].postprocess()


file_storage_instance = FileStorageWrapper()


