import os
import shutil
import imghdr
from mlapp.handlers.file_storages.file_storage_interface import FileStorageInterface
from mlapp.handlers.databases.postgres_handler import PostgresHandler
from mlapp.integrations.aml.utils.constants import OUTPUTS_FOLDER, AML_MLAPP_FOLDER
from mlapp.utils.general import read_json_file
from azureml.core import Run, Experiment
from azureml.exceptions import RunEnvironmentException
from mlapp.utils.exceptions.framework_exceptions import SkipServiceException

# postgresql queries
RUN_QUERY = '''SELECT * FROM map_run_id WHERE run_id={0}'''
UPDATE_RUN_QUERY = '''INSERT INTO map_run_id (aml_run_id, run_id, experiment_name) VALUES ({0}, {1}, {2})'''


class AmlRunStorageHandler(FileStorageInterface):
    def __init__(self, settings):
        """
        Initializes the ￿AmlRunStorageHandler with it's special connection string
        :param settings: settings from `mlapp > config.py` depending on handler type name.
        """
        super(AmlRunStorageHandler, self).__init__()
        self.temporary_storage = settings.get('temporary_storage_path')

        self.imgs_bucket = settings.get('file_store_buckets', {}).get('imgs', 'mlapp-imgs')
        self.metadata_bucket = settings.get('file_store_buckets', {}).get('metadata', 'mlapp-metadata')

        # getting azureml workspace
        try:
            self.global_run = Run.get_context(allow_offline=False)
        except RunEnvironmentException as e:
            raise SkipServiceException('Skip AmlModelStorageHandler handler')
        self.ws = self.global_run.experiment.workspace

        # creating postgres database handler
        self.db_handler = PostgresHandler(settings)

        # creating run_ids to save for postprocessing
        self.run_ids = {}

    def _get_current_run(self, run_id):
        # get azureml Run and Experiment details from postgres db
        result = self.db_handler.execute_query(RUN_QUERY, params=[run_id])
        if len(result) == 0:
            raise Exception('ERROR: run context of run_id `%s` not exists, please make sure `%s` is the right id.' %
                            (run_id, run_id))
        db_aml_run_id, db_run_id, db_experiment_name = result.pop()

        # creating Run object
        experiment = Experiment(workspace=self.ws, name=db_experiment_name)
        run = Run(experiment=experiment, run_id=db_aml_run_id)
        return run

    def download_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        This method downloads a file into your machine from aml datastore
        :param bucket_name:  name of the bucket/container, unnecessary parameter
        :param object_name: name of the object/file
        :param file_path: path to local file, unnecessary parameter
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        :return: None
        """
        # get run_id from object_name
        current_id = object_name.split('_')[0]

        # get current_id azureml Run object
        current_run: Run = self._get_current_run(current_id)

        # download file from current_run to file_path
        current_run.download_file(name=object_name, output_file_path=file_path)

    def upload_file(self, bucket_name, object_name, file_path, *args, **kwargs):
        """
        This method upload a file to the Run Object of AML
        :param bucket_name: name of the bucket/container
        :param object_name: name of the object/file
        :param file_path: path to local file
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        :return: None
        """
        run_id = kwargs.get('run_id')
        if run_id not in self.run_ids:
            self.run_ids[run_id] = True

        # upload file to Run
        self.global_run.upload_file(name=object_name, path_or_stream=file_path)

        # log images
        if bucket_name == self.imgs_bucket:
            # check if file is an image
            is_img = imghdr.what(file_path)
            if is_img is not None:
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                image_name = '_'.join(file_name.split('_')[1:])
                self.global_run.log_image(name=image_name, path=file_path)
        # log metadata
        else:
            if bucket_name == self.metadata_bucket:
                model_metadata = read_json_file(file_path)
                scores = model_metadata.get('models', {}).get('scores', None)
                if scores is not None:
                    for score_key, score_value in scores.items():
                        self.global_run.log(name=score_key, value=score_value)

    def list_files(self, bucket_name, prefix="", *args, **kwargs):
        """
        Lists files in file storage
        :param bucket_name: name of the bucket/container
        :param prefix: prefix string to search by
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        # get current_id azureml Run object
        current_run: Run = self._get_current_run(prefix)

        # return a list of files from the current run
        return current_run.get_file_names()

    def postprocess(self):
        """
        This method saves mapping from (run_id x experiment_name) to (run_id).
        :return: None
        """
        for run_id in self.run_ids.keys():
            self.db_handler.execute_query(UPDATE_RUN_QUERY, [self.global_run.id, run_id, self.global_run.experiment.name])
