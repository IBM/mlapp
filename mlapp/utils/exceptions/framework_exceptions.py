from mlapp.utils.exceptions.base_exceptions import FrameworkException


class SkipServiceException(FrameworkException):
    pass


class SkipToLocalException(FrameworkException):
    pass


class MissingConnectionException(FrameworkException):
    pass


class UnsupportedFileType(FrameworkException):
    pass


class DataFrameNotFound(FrameworkException):
    pass


class AutoMLException(FrameworkException):
    pass
