import pickle
import zipfile
import tempfile
import glob
from pathlib import Path
import numpy as np
import shutil
import os
import json
import collections
from ast import literal_eval
import collections.abc
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def recursive_dict_update(d, u):
    if hasattr(u, 'items'):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = recursive_dict_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d


def search_key_recursive( d, key):
    if hasattr(d, 'items'):
        for k, v in d.items():
            if key == k:
                yield v
            if isinstance(v, dict):
                for result in search_key_recursive(v, key):
                    yield result
            elif isinstance(v, list):
                for i in v:
                    for result in search_key_recursive(i, key):
                        yield result


def __convert__(data):
    if isinstance(data, str):
        return str(data)
    elif isinstance(data, collections.abc.Mapping):
        return dict(map(__convert__, data.items()))
    elif isinstance(data, collections.abc.Iterable):
        return type(data)(map(__convert__, data))
    else:
        return data


def read_json_file(path, file_name=None):
    if file_name is None:
        with open(path, "r") as f:
            data = json.load(f)
    else:
        with open(os.path.join(path, file_name), "r") as f:
            data = json.load(f)
    return __convert__(data)


def load_json_unicode(body):
    data = json.loads(literal_eval(body))
    return data


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_project_root():
    # change this path accordingly when moving this file
    return Path(__file__).parent.parent.parent


def load_pickle_to_object(path):
    obj = pickle.load(open(path, 'rb'))
    return obj


def save_object_to_pickle(model, path):
    pickle.dump(model, open(path, 'wb'))


def extract_zip_file(file_path):
    try:
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(file_path)
        zip_ref.close()
    except Exception as e:
        raise RuntimeError('failed to unzip file: ' + str(e))


def compress_to_zip_file(file_path):
    try:
        zip_ref = zipfile.ZipFile(file_path + ".zip", 'w', zipfile.ZIP_DEFLATED)
        zip_ref.write(file_path)
        zip_ref.close()
    except Exception as e:
        raise RuntimeError('failed to compress file: ' + str(e))


def save_object_tempfile(file_name, obj):
    tempdir = tempfile.mkdtemp()
    file_path = os.path.join(tempdir, file_name + '.pkl')
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)
    return file_path


def get_compressed_file_and_remove_uncompressed(file_path):
    compress_to_zip_file(file_path)
    os.remove(file_path)


def compress_folder_to_zip(file_path):
    shutil.make_archive(file_path, 'zip', file_path)
    shutil.rmtree(file_path)


def uncompress_zip_to_folder(file_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.mkdir(extract_dir)
        shutil.unpack_archive(file_path, extract_dir=extract_dir)
    os.remove(file_path)


def save_object_as_json(obj, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(obj, outfile, cls=NumpyEncoder)


def is_jsonable(obj):
    """
    checks if object is json serializable.
    :param obj: python object
    :return: bool
    """
    try:
        json.dumps(obj,cls=NumpyEncoder)
        return True
    except:
        return False


def update_json_file(new_data, file_path):
    """
    This function reads existing data from json file and append it with the new data and save the file.
    :param new_data: dictionary - new data to append
    :param file_path: string - path to file
    :return: None
    """

    # read existing data from json file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # updates the json file data
    data.update(new_data)

    # write new data to the json file
    with open(file_path, 'w') as f:
        json.dump(data, f)


def validate_json(d, warnning_message=None):
    """
    This function get an object and check if its a vaild json
    and deletes everything that not json serializable
    :param d: python dictionary
    :param warnning_message: str
    :return: None (void)
    """
    if not isinstance(d, dict):
        raise Exception('Error: object is not dictionary')

    keys_to_delete = []
    for key,value in d.items():

        # if d[key] is also a dictionary call recursive check
        if isinstance(d[key], dict):
            validate_json(d[key], warnning_message)

        if not is_jsonable(value):
            keys_to_delete.append(key)

    for key in keys_to_delete:
        if key in d:
            if warnning_message is not None:
                print("Warnning: " + warnning_message % (str(key)))
            del d[key]


def save_figure_to_png(figure, path, file_name):
    try:
        figure.savefig(path)
    except:
        try:
            FigureCanvas(figure)
            figure.savefig(path)
        except Exception as e:
            raise Exception('Failed to save %s plot image.' % (str(file_name)))


def read_files_names_from_local_storage(prefix, local_storage_path):
    '''
    Reads all files starting with prefix from local storage.
    @param prefix: str, filename name prefix to look for
    @param local_storage_path: str, local path to look for files.
    @return: list of strings, file names.
    '''
    return list(map(lambda f: os.path.basename(f), glob.glob(os.path.join(local_storage_path, prefix + "_*"))))


def create_tempdir(name):
    return tempfile.mkdtemp(prefix=name)


def delete_directory_with_all_contents(path):
    shutil.rmtree(path=path)


def create_directory(directory_name, path=''):
    # concat the path and directory name.
    full_directory_path = os.path.join(path, directory_name)
    full_directory_path = os.path.join(os.getcwd(), full_directory_path)

    # checks if directory already exists, if not then creating it.
    if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)
        return full_directory_path
    else:
        raise Exception('ERROR: path %s already exists.' % str(full_directory_path))

