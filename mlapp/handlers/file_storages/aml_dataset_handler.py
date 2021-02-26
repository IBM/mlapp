from mlapp.handlers.file_storages.file_storage_interface import FileStorageInterface
from azureml.core.dataset import Dataset
from azureml.core import Run
import pandas as pd
from azureml.exceptions import RunEnvironmentException
from mlapp.utils.exceptions.framework_exceptions import SkipServiceException


class AmlDatasetHandler(FileStorageInterface):
    def __init__(self, settings):
        super(AmlDatasetHandler, self).__init__()

    def download_file(self, dataset_name, dataset_type='pandas', *args, **kwargs):
        """
        Downloads file from file storage
        :param dataset_name: name of the dataset
        :param dataset_type: name of the dataset type
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        try:
            run = Run.get_context(allow_offline=False)
        except RunEnvironmentException as e:
            raise SkipServiceException('Skip AmlModelStorageHandler handler')
        ws = run.experiment.workspace
        ds = Dataset.get_by_name(workspace=ws, name=dataset_name)  # Get a Dataset by name
        if dataset_type == 'spark':
            df = ds.to_spark_dataframe()  # Load a Tabular Dataset into pandas DataFrame
        else:
            df = ds.to_pandas_dataframe()  # Load a Tabular Dataset into pandas DataFrame
        return df

    def upload_file(self, dataset: pd.DataFrame, dataset_name: str, dataset_type: str='pandas', *args, **kwargs):
        """
        Uploads file to file storage
        :param dataset: dataset object
        :param dataset_name: name of the dataset
        :param dataset_type: name of the dataset type
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        run = Run.get_context()
        ws = run.experiment.workspace
        if dataset_type == 'pandas':
            ds = Dataset.from_pandas_dataframe(dataframe=dataset)  # Load a Tabular Dataset into pandas DataFrame
        else:
            raise Exception("ERROR: only pandas DataFrame is supported ath the moment.")

        ds.register(ws, name=dataset_name)

    def list_files(self, bucket_name, prefix="", *args, **kwargs):
        """
        Lists files in file storage
        :param bucket_name: name of the bucket/container
        :param prefix: prefix string to search by
        :param args: other arguments containing additional information
        :param kwargs: other keyword arguments containing additional information
        """
        raise Exception("ERROR: Azure Machine Lreaning Dataset class not supported this method.")
