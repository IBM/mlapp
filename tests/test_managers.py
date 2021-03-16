import unittest
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from mlapp.utils.general import get_project_root
from mlapp.managers import ModelManager, DataManager#, _UserManager
from mlapp.managers.io_manager import IOManager
from mlapp.utils.automl import AutoMLResults
from pyspark.ml.regression import LinearRegression as spark_lr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

os.chdir(get_project_root())

#import random

class TestAssets(unittest.TestCase):

    def _set_up_automl_results(img):
        best_estimator = None
        feature_selection = None
        train_predicted = np.random.rand(10)
        test_predicted= np.random.rand(10)
        selected_features = []
        best_model = None
        intercept = np.random.rand(1)[0]
        coefficients= np.random.rand(2)
        cv_score= np.random.rand(1)[0]
        all_cv_scores= pd.DataFrame(np.random.rand(5,5), columns=[x for x in 'abcde'])
        figures= {'img_1': img, 'img_2': img}

        automl_reults = AutoMLResults()
        automl_reults.add_cv_run(mean_squared_error, best_estimator, feature_selection, train_predicted, test_predicted,
                   selected_features, best_model, intercept, coefficients, cv_score, all_cv_scores, figures)
        
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
        dm._get_manager_type() #return 'data'  
        dm._output_manager.add_dataframe('features', self.df)
         # RE-Initialize data manager:
        dm = DataManager(config={}, _input=dm._output_manager, _output=IOManager(), run_id='')
        self.assertEquals(dm._input_manager.get_dataframe('features').shape, self.df.shape, "df shape unmached")

    def test_model_manager(self):
        preds = self.df.sample(10) 
        # Initialize model manager:
        mm = ModelManager(config={}, _input=IOManager(), _output=IOManager(), run_id='')
        mm._get_manager_type() #return 'models'
        mm._output_manager.add_dataframe('predictions', preds, 'target')
        # RE-Initialize model manager:
        mm = ModelManager(config={}, _input=mm._output_manager, _output=IOManager(), run_id='')
        self.assertEquals(mm.get_dataframe('predictions').shape, preds.shape, "predictions shape unmached")
    
    def test_user_manager(self):
        # Initialize model manager:
        um = ModelManager(config={}, _input=IOManager(), _output=IOManager(), run_id='')
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
        d={um._get_manager_type():{}}
        [d[um._get_manager_type()].update(obj) for obj in [um._output_manager.objects[k][um._get_manager_type()] for k in list(um._output_manager.objects.keys())]]
        um._output_manager.objects = d

        # RE-Initialize model manager:
        um = ModelManager(config={}, _input=um._output_manager, _output=IOManager(), run_id='')
        #self.assertEquals(um._get_all_metadata(), {'models': {'key': 1}}, 'metadata unmatch')
        um.get_object('my_obj')
        um._get_all_objects()
        um.get_metadata('key', default_value=None)
        um.get_dataframe('features') 
        um.get_dataframe('features_table') 
        um.get_automl_result() 
    
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

        #### GET Functions
        self.assertEquals(iom.get_dataframe('features').shape, self.df.shape, "df shape unmached")
        iom.get_tables()
        objects = iom.get_objects()
        iom.get_objects_value('my_obj')
        iom.get_images_files()
        iom.get_metadata()
        iom.get_metadata_value(managerType)
        #iom.get_all_values() does not pass with error => NotImplementedError: TransformNode instances can not be copied. Consider using frozen() instead. MC
        iom.get_all_keys()
        for cat in ['dataframes', 'tables', 'metadata', 'images', 'objects', 'ids', 'new_structure']:
            k = iom.get_all_keys_per_cat(cat)
            v = iom.get_all_values_per_category(cat)
            for k_, v_ in zip(k, v.values()):
                self.assertEquals(type(iom.structure[cat][k_]), type(v_))
        iom.search_key_value('key')
        #iom.set_nested_value(self, dictionary, category, key, value)
       
if __name__== "__main__":
    b = ''
    unittest.main()
