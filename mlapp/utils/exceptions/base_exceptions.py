"""
Base Exception
MLApp Exception - inherit from Base Exception
"""


class MLAppBaseException(Exception):
    def __init__(self, message):
        self.message = message


class FrameworkException(MLAppBaseException):
    def __init__(self, message=None):
        if message is not None and isinstance(message, str):
            self.message = message

    def __str__(self):
        return "[ML APP ERROR] %s\n" % str(self.message)


class UserException(MLAppBaseException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[USER ERROR] %s\n" % str(self.message)


class FlowManagerException(UserException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[FLOW MANAGER ERROR] %s\n" % str(self.message)


class DataManagerException(UserException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[DATA MANAGER ERROR] %s\n" % str(self.message)


class ModelManagerException(UserException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[MODEL MANAGER ERROR] %s\n" % str(self.message)


class JobManagerException(UserException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[JOB MANAGER ERROR] %s\n" % str(self.message)


class PipelineManagerException(UserException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[PIPELINE MANAGER ERROR] %s\n" % str(self.message)


class EnvironmentException(UserException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[ENVIRONMENT ERROR] %s\n" % str(self.message)


class IoManagerException(FlowManagerException, DataManagerException, ModelManagerException, JobManagerException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[IO MANAGER ERROR] %s\n" % str(self.message)


class ConfigError(FlowManagerException, DataManagerException, ModelManagerException, JobManagerException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[CONFIG ERROR] %s\n" % str(self.message)


class ConfigKeyError(ConfigError):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[KEY ERROR] %s\n" % str(self.message)


class ConfigValueError(ConfigError):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "[VALUE ERROR] %s\n" % str(self.message)
