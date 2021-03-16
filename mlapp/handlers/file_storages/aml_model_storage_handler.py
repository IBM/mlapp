import os
import logging
import shutil
import imghdr
import glob
from mlapp.handlers.file_storages.file_storage_interface import FileStorageInterface
from mlapp.integrations.aml.utils.run_class import get_model_register_name
from mlapp.integrations.aml.utils.constants import OUTPUTS_FOLDER, AML_MLAPP_FOLDER
from mlapp.utils.general import read_json_file
from azureml.core import Model, Run
from azureml.exceptions import RunEnvironmentException
from mlapp.utils.exceptions.framework_exceptions import SkipServiceException


class AmlModelStorageHandler(FileStorageInterface):
    def __init__(self, settings):
        """
        Initializes the ￿AmlModelStorageHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(AmlModelStorageHandler, self).__init__()

        self.registered_models = {}
        self.downloaded_files = {}
        self.temporary_storage = settings.get('temporary_storage_path')

        self.imgs_bucket = settings.get('file_store_buckets', {}).get('imgs', 'mlapp-imgs')
        self.metadata_bucket = settings.get('file_store_buckets', {}).get('metadata', 'mlapp-metadata')

        try:
            self.run = Run.get_context(allow_offline=False)
        except RunEnvironmentException as e:
            raise SkipServiceException('Skip AmlModelStorageHandler handler')
        self.ws = self.run.experiment.workspace

        # create AML outputs folder
        os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
        os.makedirs(os.path.join(OUTPUTS_FOLDER, AML_MLAPP_FOLDER), exist_ok=True)

    def _register(self, run_id, asset_name, asset_label=None):
        """
        This method register a new model as a AML reference.
        :param run_id: run's identifier
        :param asset_name: name of asset of the current run
        :param asset_label: name of label of the current run
        :return: None
        """
        tags = {
            "run_id": run_id,
            "asset_name": asset_name
        }

        if asset_label is not None:
            tags["asset_label"] = asset_label

        register_path = os.path.join(OUTPUTS_FOLDER, AML_MLAPP_FOLDER)
        Model.register(
            self.ws,
            model_path=register_path,
            model_name=get_model_register_name(run_id),
            tags=tags,
            description=asset_name)

    def download_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        This method downloads a file into your machine from AML datastore
        :param bucket_name:  name of the bucket/container, unnecessary parameter
        :param object_name: name of the object/file
        :param file_path: path to local file, unnecessary parameter
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        :return: None
        """
        # get run_id from object_name
        current_id = object_name.split('_')[0]

        if current_id not in self.downloaded_files.keys() or not self.downloaded_files[current_id]:
            # load registered models
            model_name = get_model_register_name(current_id)
            model = Model(self.ws, name=model_name)
            model.download(target_dir=current_id, exist_ok=True)

            # mark current_id as registered
            self.downloaded_files[current_id] = {
                "path": current_id
            }

        # move file to AML_MLAPP_FOLDER folder from current_id folder
        if os.path.exists(os.path.join(current_id, AML_MLAPP_FOLDER, object_name)):
            shutil.move(os.path.join(current_id, AML_MLAPP_FOLDER, object_name), self.temporary_storage)

    def upload_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        This method move a file to the registered folder of AML
        :param bucket_name:  name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        :return: None
        """
        # copy file from file_path to register_path. file deletion is handled by job manager
        register_path = os.path.join(OUTPUTS_FOLDER, AML_MLAPP_FOLDER)
        output_path = os.path.join(register_path, os.path.basename(file_path))
        shutil.copy(file_path, output_path)

        run_id = kwargs.get('run_id')
        if run_id not in self.registered_models.keys():
            # save run_id parameters.
            self.registered_models[run_id] = {
                "asset_name": kwargs.get('asset_name'),
                "asset_label": kwargs.get('asset_label')
            }

        # log images
        if bucket_name == self.imgs_bucket:
            # check if file is an image
            is_img = imghdr.what(file_path)
            if is_img is not None:
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                image_name = '_'.join(file_name.split('_')[1:])
                self.run.log_image(name=image_name, path=output_path)
        # log metadata
        else:
            if bucket_name == self.metadata_bucket:
                model_metadata = read_json_file(file_path)
                scores = model_metadata.get('models', {}).get('scores', None)
                if scores is not None:
                    for score_key, score_value in scores.items():
                        self.run.log(name=score_key, value=score_value)

    def list_files(self, bucket_name, prefix="", *args, **kwargs):
        """
        Lists files in file storage
        :param bucket_name: name of the bucket/container
        :param prefix: prefix string to search by
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        try:
            # get run_id from object_name
            current_id = prefix

            if current_id not in self.downloaded_files.keys():
                # load registered models
                model_name = get_model_register_name(current_id)
                model = Model(self.ws, name=model_name)
                model.download(target_dir=current_id, exist_ok=True)

                # mark current_id as registered
                self.downloaded_files[current_id] = {
                    "path": current_id
                }

            # get files from current_id folder and temporary_storage folder
            list_files = os.listdir(os.path.join(current_id, AML_MLAPP_FOLDER))
            list_files_extra = list(
                map(lambda f: os.path.basename(f), glob.glob(os.path.join(self.temporary_storage, current_id + "_*"))))
            list_files.extend(list_files_extra)
            return list_files

        except Exception as e:
            logging.error(e)
            raise e

    def _clear(self):
        """
        This method deletes all used and unnecessary files.
        :return: None
        """
        # deletes all unnecessary files
        for run_id_to_del, metadata in self.downloaded_files.items():
            path = metadata.get("path", None)
            if path is not None and os.path.isdir(path):
                shutil.rmtree(path)

        # clear self.downloaded_files
        self.downloaded_files = {}

    def postprocess(self):
        """
        This method register and clear unnecessary files.
        :return: None
        """

        # register models
        for run_id, values in self.registered_models.items():
            asset_name = values.get('asset_name')
            asset_label = values.get('asset_label')
            self._register(run_id, asset_name, asset_label)

        # clear all files after registering
        self._clear()
