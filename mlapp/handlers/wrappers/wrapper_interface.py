from abc import ABCMeta, abstractmethod
from mlapp.config import settings
from mlapp.utils.exceptions.framework_exceptions import SkipServiceException


class WrapperInterface:
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Constructor for he Wrapper Interface.
        Will hold the handlers set up in the environment in the `self._handlers` in a dictionary by key name.
        Will hold a list containing the key names of handlers set up as the main ones.
        """
        self._handlers = {}
        self._main_handlers = []
        self.initialized = False

    @abstractmethod
    def init(self, handler_type):
        """
        Initialization, should be called once only
        Populates the `self._handlers` and `self_main_handlers` variables depending on the set environment
        :param handler_type: used for filtering services by the handler type
        """
        if not self.initialized:
            for service_name in settings.get('services', []):
                service_item = settings['services'][service_name]
                if 'type' not in service_item:
                    raise Exception("'{}' service is missing 'type' key, must be filled in config.py with"
                                    " the one of the following: database/file_storage/database/spark".format(service_name))
                if service_item['type'] == handler_type:
                    try:
                        self._handlers[service_name] = service_item['handler'](service_item.get('settings', {}))

                        # set it as main
                        if service_item.get('main', False):
                            self._main_handlers.append(service_name)

                    except SkipServiceException as e:
                        pass  # skipping this service
                    except Exception as e:
                        if service_item['handler'] is None:
                            raise Exception("'{}' service of type '{}' is missing a python library installation."
                                            .format(service_name, service_item.get('type')))
                        else:
                            raise e
            self.initialized = True

    def get(self, handler_name):
        """
        Get the handler instance by name
        :param handler_name: handler name string
        :return: Handler Instance
        """
        return self._handlers.get(handler_name)

    def empty(self):
        """
        Checks if there are configured handlers as "main"
        """
        return len(self._main_handlers) == 0
