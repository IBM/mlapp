import pandas as pd
import copy
from mlapp.utils.general import recursive_dict_update, search_key_recursive
from mlapp.utils.exceptions.base_exceptions import IoManagerException
from matplotlib.figure import Figure
try:
    import plotly.graph_objects as go
except:
    go = None


class IOManager(object):
    def __init__(self, ):
        """
        IO Manager constructor defines the structures of input or output that are common in an asset.
        e.g. dataframes, metadata (dictionaries) , images, objects
        """

        self.dataframes = {}
        self.tables = {}
        self.metadata = {}
        self.images = {}
        self.objects = {}
        # self.config = {}
        self.ids = {}

        self.structure = {
            'dataframes': self.dataframes,
            'tables': self.tables,
            'metadata': self.metadata,
            'images': self.images,
            'objects': self.objects,
            'ids': self.ids,

        }
        self.to_store = {
            'images': False,
            'objects': False,
            'dataframes': False,
            'tables': False,
            'metadata': False,
            # 'config': False
        }

    #### SET Functions

    def add_dataframe(self, name, data, to_table=None):
        """
        Adds a dataframe to the dictionary of dataframes
        :param name: dataframe name
        :param data: dataframe object
        :param to_table: table name in physical database, to be saved to
        :return:
        """
        if not self.dataframes.get(name, None) is None:
            self.dataframes[name] = pd.concat([self.dataframes[name], data])
        else:
            self.dataframes[name] = data
        if to_table:
            self.tables[name] = to_table

    def set_objects_value(self, objects_category, key, value):
        """
         Adds an object to the dictionary of objects, e.g. model pkl
        :param objects_category: category name defined by user, e.g. "models"
        :param key: object name
        :param value: object value to update
        :return:
        """
        if self.objects.get(objects_category, []):

            self.objects[objects_category][key] = value
        else:
            self.objects[objects_category] = {key: value}
        self.to_store['objects'] = True

    def add_objects(self, objects_category, objects_dict):
        """
        Replaces/set  dictionary of objects under certain category
        :param objects_category: category name
        :param objects_dict: dict of objects
        :return:
        """
        if not isinstance(objects_dict, dict):
            raise IoManagerException(
                'IOManager:add_objects() received non dictionary as input. please pass dictionary only')

        if self.objects.get(objects_category, []):
            self.objects[objects_category] = recursive_dict_update(self.objects[objects_category], objects_dict)
        else:
            self.objects[objects_category] = objects_dict

    def set_analysis_metadata_value(self, analysis_metadata_name, key, value):
        """

        :param analysis_metadata_name: name of the metadata dictionary
        :param key: internal key in the dictionary
        :param value: internal value in the dictionary
        :return:
        """

        if self.metadata.get(analysis_metadata_name, []):
            self.metadata[analysis_metadata_name][key] = value
        else:
            self.metadata[analysis_metadata_name] = {key: value}

    def add_analysis_metadata(self, analysis_metadata_name, metadata_dict):
        """
        :param analysis_metadata_name: name of the metadata dictionary
        :param metadata_dict:the metadata dictionary
        :return:
        """
        if not isinstance(metadata_dict, dict):
            raise IoManagerException(
                'IOManager:set_analysis_metadata() received non dictionary as input. please pass dictionary only')

        if self.metadata.get(analysis_metadata_name, []):
            self.metadata[analysis_metadata_name] = recursive_dict_update(self.metadata[analysis_metadata_name],
                                                                          metadata_dict)
        else:
            self.metadata[analysis_metadata_name] = metadata_dict

    def add_images(self, imgs):
        """
        :param imgs: dictionary of images with image names and objects
        :return:
        """
        if isinstance(imgs, dict):
            for key in imgs:
                if isinstance(imgs[key], list):
                    for index, figure in enumerate(imgs[key]):
                        self.add_image(key + '_' + str(index), figure)
                else:
                    self.add_image(key, imgs[key])
            if len(imgs):
                self.to_store['images'] = True
        else:
            raise IoManagerException("Error in `add_images`! Send a dictionary where: \n"
                            " - Keys are the names of the image/s \n"
                            " - Values are `matplotlib` Figures, or a list of `matplotlib` Figures.")

    def add_image(self, name, figure):
        """
        :param name: image name
        :param figure:  image object
        :return:
        """
        if isinstance(figure, Figure) or (go and isinstance(figure, go.Figure)):
            self.structure['images'][name] = figure
            self.to_store['images'] = True
        else:
            raise IoManagerException(f"{name} is not in the right format. " 
                                     "ML App supports only figures from the matplotlib or plotly.js library.")

    # def add_config(self, config):
    #     self.config = config
    #
    # def set_config_value(self, key, value):
    #     self.set_nested_value(self.config, '', key, value)

    def add_key(self, key, value):
        """
        Adding a new structure in the IO manager structure
        :param key: name of the structure
        :param value: value to be set
        :return:
        """
        self.structure[key] = value

    #### GET Functions

    def get_dataframes(self):
        """
        :return: returns dictionary of dataframes
        """
        return self.dataframes

    def get_tables(self):
        """
        :return: dictionary of dataframe name:table name
        """
        return self.tables

    def get_dataframe(self, name, default=None):
        """
        :param name: name of the dataframe to be retrieved
        :param default: default value in case it doesn't exist
        :return: dataframe or default value
        """
        return self.dataframes.get(name, default)

    def get_objects(self):
        """
        :return: returns dictionary of objects
        """
        return self.objects

    def get_objects_value(self, key, default=None):
        """

        :param key: object key to be returned
        :param default:
        :return:
        """
        return self.objects.get(key, default)

    def get_images_files(self):
        """
        :return: a dictionary of images
        """
        return self.images

    # def get_config(self):
    #     return copy.deepcopy(self.config)

    def get_metadata(self):
        """
        :return: dictionary of metadata
        """
        #return copy.deepcopy(self.metadata)
        return self.metadata

    def get_metadata_value(self, analysis_metadata_name, default=None):
        """

        :param analysis_metadata_name: name of metadataa dictionary
        :param default: default value in case it doesn't exist
        :return: dictionary
        """
        return self.metadata.get(analysis_metadata_name, default)

    def get_all_values(self):
        """

        :return: the IO Manager structure and its content
        """
        return copy.deepcopy(self.structure)

    def get_all_keys(self):
        """

        :return: the IO Manager structure
        """
        return list(self.structure.keys())

    def get_all_keys_per_cat(self, category):
        """
        :param category: the IO Manager specific category and all its values (e.g. objects)
        :return:
        """
        if category in self.structure.keys():
            return list(self.structure[category].keys())
        else:
            raise IoManagerException("Unknown nested category" + category + " in IO_Manager."
                                                                   "Please insert one of the categories supported:" + ", ".join(
                list(self.structure.keys())))

    def get_all_values_per_category(self, category):
        """
        :param category: the structure category name ( e.g. "objects")
        :return: the values in that specific category, if exists
        """
        if category in self.structure.keys():
            return self.structure[category]
        else:
            raise IoManagerException("Unknown nested category" + category + " in IO_Manager."
                                                                   "Please insert one of the categories supported:" + ", ".join(
                list(self.structure.keys())))


    def search_key_value(self,key):
        """
        searches a key in the whole structure
        :param key: key to search
        :return: all values found, in a list
        """
        results = []
        for res in search_key_recursive(self.structure,key):
            results.append(res)
        return results

    #########################
    @staticmethod
    def prepare_obj_for_json(to_json):
        return copy.deepcopy(to_json)

    # def prepare_models_objects_for_package(self):
    #     """
    #     Preparing result for pickle
    #     :return: result for pickle
    #     """
    #     response = copy.deepcopy(self.results['analysis_results'])
    #     for key in self.results['models_objects']:
    #         try:
    #             response[key] = copy.deepcopy(self.results['models_objects'][key])
    #         except AttributeError as e:
    #             print("Error copying object `{}`. Reason: ".format(key), str(e))
    #             print("Trying to copy using object's copy function..")
    #             if isinstance(self.results['models_objects'][key], list):
    #                 response[key] = [v.copy() for v in self.results['models_objects'][key]]
    #             if isinstance(self.results['models_objects'][key], dict):
    #                 response[key] = {k: v.copy() for (k, v) in self.results['models_objects'][key].items()}
    #             else:
    #                 response[key] = self.results['models_objects'][key].copy()
    #             print("Copy Success!")
    #     return response

    def set_nested_value(self, dictionary, category, key, value):
        if key in dictionary[category].keys():
            dictionary[category][key] = value
        else:
            for k, v in dictionary[category].items():
                self.set_nested_value(dictionary[category], k, key, value)

    # def get_value(self, dictionary,category, key):
    #     if category in dictionary.keys():
    #         if key in dictionary[category]:
    #             return copy.deepcopy(dictionary[category][key])
    #
    #         else:
    #             raise exception("Unknown key " + key + " in "+category+ "dictionary." +
    #                             "Please insert the key into the right dictionary before calling it ")
    #     else:
    #         raise exception("Unknown nested category: "+category)
