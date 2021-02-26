from environs import Env
from os.path import join

EMPTY_STRING = ""
ZERO = 0


class EnvironmentLoader:
    """ This class is responsible for load and filter environment variables. """

    @staticmethod
    def load(filename, path=''):
        """
        This function read env file
        :param filename: String, env filename
        :param path: String, env file path (Optional)
        :return: Env object
        """
        try:
            env = Env()
            full_path = join(path, filename)
            env.read_env(full_path, recurse=False, override=True)

            return env
        except ValueError:
            return Env()
        except Exception as e:
            raise Exception('Error in EnvironmentLoader, load method ' + str(e))

    @staticmethod
    def create_services(env, services_dict, available_services):
        """
        This file creates the services based on the `.env` file and environment variables.
        """
        services_settings = {}

        for service_name, service_type in services_dict.items():
            services_settings[service_name] = available_services(env)[service_type](service_name)

        return services_settings
