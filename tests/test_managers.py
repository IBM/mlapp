import unittest
import pandas as pd
import os
import numpy as np
from mlapp.utils.metrics.pandas import regression
from mlapp.utils.general import get_project_root
from mlapp.managers import ModelManager, DataManager
from mlapp.managers.io_manager import IOManager
from mlapp.utils.automl import AutoMLResults
from pyspark.ml.regression import LinearRegression as spark_lr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

os.chdir(get_project_root())


class TestManagers(unittest.TestCase):
    @staticmethod
    def _set_up_automl_results(img):
        best_estimator = None
        feature_selection = None
        train_predicted = np.random.rand(10)
        y_train = np.random.rand(10)
        test_predicted = np.random.rand(10)
        y_test = np.random.rand(10)
        selected_features = []
        best_model = None
        intercept = np.random.rand(1)[0]
        coefficients = np.random.rand(2)
        cv_score = np.random.rand(1)[0]
        all_cv_scores = pd.DataFrame(np.random.rand(5,5), columns=[x for x in 'abcde'])
        figures = {'img_1': img, 'img_2': img}

        automl_reults = AutoMLResults()
        automl_reults.add_cv_run(regression, best_estimator, feature_selection, train_predicted, test_predicted,
                                 selected_features, best_model, intercept, coefficients, cv_score, all_cv_scores,
                                 figures, y_train=y_train, y_test=y_test)
        
        return automl_reults
    
    @classmethod
    def setUpClass(cls):
        path = './data/diabetes.csv'
        df = pd.read_csv(path)
        cls.df = df

        lr_model = LinearRegression()
        cls.obj = lr_model
        cls.spark_obj = spark_lr
        cls.img = plt.figure(figsize=(12, 10), dpi=80)

        cls.automl_results = cls._set_up_automl_results(cls.img)

    def setUp(self):
        pass

    def test_data_manager(self):
        # Initialize data manager:
        dm = DataManager(config={}, _input=IOManager(), _output=IOManager(), run_id='')
        self.assertEqual(dm._get_manager_type(), "data", "wrong manager type returned")   
        dm._output_manager.add_dataframe('features', self.df)
        # RE-Initialize data manager:
        dm = DataManager(config={}, _input=dm._output_manager, _output=IOManager(), run_id='')
        self.assertEqual(dm._input_manager.get_dataframe('features').shape, self.df.shape, "df shape unmatched")

    def test_model_manager(self):
        preds = self.df.sample(10) 
        # Initialize model manager:
        mm = ModelManager(config={}, _input=IOManager(), _output=IOManager(), run_id='')
        self.assertEqual(mm._get_manager_type(), "models", "wrong manager type returned")   
        mm._output_manager.add_dataframe('predictions', preds, 'target')
        # RE-Initialize model manager:
        mm = ModelManager(config={}, _input=mm._output_manager, _output=IOManager(), run_id='')
        self.assertEqual(mm.get_dataframe('predictions').shape, preds.shape, "predictions shape unmatched")
    
    def test_user_manager(self):
        # Initialize model manager:
        um = ModelManager(config={}, _input=IOManager(), _output=IOManager(), run_id='')
        self.assertEqual(um._get_manager_type(), "models", "wrong manager type returned")   
        um.save_metadata('key', 1)
        um.save_object('my_obj', self.obj)
        um.save_object('my_pyspark_object', self.spark_obj, obj_type='pyspark')
        um.save_object('my_tensorflow_object', self.obj, obj_type='tensorflow')
        um.save_object('my_keras_object', self.obj, obj_type='keras')
        um.save_object('my_pytorch_object', self.obj, obj_type='pytorch')
        um.save_images({'img_1': self.img, 'img_2': self.img})
        um.save_image('my_image', self.img)
        um.save_dataframe('features', self.df, to_table=None)
        um.save_dataframe('features_table', self.df, to_table='my_table')
        um.save_automl_result(self.automl_results, obj_type='pkl')
        
        # Mock jobmanager:
        d = {um._get_manager_type():{}}
        [d[um._get_manager_type()].update(obj) for obj in [um._output_manager.objects[k][um._get_manager_type()] for
                                                           k in list(um._output_manager.objects.keys())]]
        um._output_manager.objects = d

        # RE-Initialize model manager:
        um = ModelManager(config={}, _input=um._output_manager, _output=IOManager(), run_id='')
        self.assertEqual(len(um._get_all_metadata()[um._get_manager_type()].keys()), 9, 'metadata unmatch')
        self.assertEqual(type(um.get_object('my_obj')).__name__, 'LinearRegression', "class name unmatched")
        self.assertEqual(len(um._get_all_objects()[um._get_manager_type()].keys()), 6, 'number of keys unmatch')
        self.assertEqual(um.get_metadata('key', default_value=None), 1, 'metadata unmatch')
        self.assertEqual(um.get_dataframe('features').shape, self.df.shape, "df shape unmatched")
        self.assertEqual(um.get_dataframe('features_table').shape, self.df.shape, "df shape unmatched")
        self.assertEqual(type(um.get_automl_result()).__name__, 'AutoMLResults', "class name unmatched")
    
    def test_io_manager(self):
        managerType = 'IOManager'
        iom = IOManager()
        iom.add_dataframe('features', self.df, to_table=None)
        iom.add_dataframe('features_table', self.df, to_table='my_table')
        iom.set_objects_value(managerType, 'my_obj', self.obj)
        iom.add_objects(managerType, {'my_obj': self.obj})
        iom.set_analysis_metadata_value(managerType, 'key', 1)
        iom.add_analysis_metadata(managerType, {'key_2': 2, 'key_3': 3})
        iom.add_images({'img_1': self.img, 'img_2': self.img})
        iom.add_image('img', self.img)
        iom.add_key('new_structure', {})

        # Mock jobmanager:
        iom.objects = iom.objects[managerType]

        #### GET Functions
        self.assertEqual(iom.get_dataframe('features').shape, self.df.shape, "df shape unmatched")
        self.assertEqual(iom.get_tables(), {'features_table': 'my_table'})
        self.assertEqual(type(iom.get_objects()['my_obj']).__name__ ,'LinearRegression', 'class name unmatch')
        self.assertEqual(type(iom.get_objects_value('my_obj')).__name__ ,'LinearRegression', 'class name unmatch')
        self.assertEqual(type(iom.get_images_files()['img']).__name__, 'Figure', 'class name unmatch')
        self.assertEqual(len(iom.get_metadata()[managerType].keys()), 3, "number of keys unmatch")
        self.assertEqual(len(iom.get_metadata_value(managerType).keys()), 3, "number of keys unmatch")

        # iom.get_all_values() does not pass with error =>
        # NotImplementedError: TransformNode instances can not be copied. Consider using frozen() instead. MC
        iom.get_all_keys()
        for cat in ['dataframes', 'tables', 'metadata', 'images', 'objects', 'ids', 'new_structure']:
            k = iom.get_all_keys_per_cat(cat)
            v = iom.get_all_values_per_category(cat)
            for k_, v_ in zip(k, v.values()):
                self.assertEqual(type(iom.structure[cat][k_]), type(v_))
        self.assertEqual(iom.search_key_value('key'), [1], "value unmatched")
        # iom.set_nested_value(self, dictionary, category, key, value)


if __name__== "__main__":
    unittest.main()
