from mlapp.managers import ModelManager
from mlapp.managers import DataManager
from mlapp.managers.io_manager import IOManager
from pyspark.ml.regression import LinearRegression as spark_lr
import random


if __name__== "__main__":
    pass

    '''
    # imports and test data:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression

    ###########################################
    #          generate sample data           #
    ###########################################

    path = '../../data/diabetes.csv'
    df = pd.read_csv(path)
    df_name = path.split('/')[-1]

    X, y_true = split_X_and_y(df, 'target')
    X_train, y_train, X_test, y_test, index_train, index_test = split_train_test_with_prop(X, y_true, 0.8)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    train_y_predict = lr_model.predict(X_train)
    test_y_predict = lr_model.predict(X_test)

    scores = get_model_accuracy_for_regression(train_y_predict, y_train, test_y_predict, y_test)


    obj = lr_model
    spark_obj = spark_lr

    # img = plt.figure(figsize=(12, 10), dpi=80)
    img = X_train.plot().figure
    img.show()
    plt.show()

    best_model_name = lr_model.__class__.__name__
    best_feature_selection = 'best_feature_selection'

    selected_features = random.choices(list(X_train.columns), k=4)
    coefficients = lr_model.coef_
    feature_selection_method = 'feature_selection_method'
    constant = lr_model.intercept_
    missing_values = {x: np.mean(X_train[x]) for x in selected_features}

    analysis_results_dict = {
        'scores': scores,
        'selected_features': selected_features,
        'coefficients': coefficients,
        'feature_selection_method': feature_selection_method,
        'constant': constant
        }

    analysis_results_dict_2 = {
        'scores': {'val':4, 'val2':5},
        'selected_features': selected_features,
        'coefficients': coefficients,
        'feature_selection_method': feature_selection_method,
        'constant': constant
        }


    print('''
    ###########################################
    #          data manager tests             #
    ###########################################
    ''')

    # Initialize data manager:
    dm = DataManager(config={}, _input=IOManager(), _output=IOManager())

    # saves:
    dm.save_dataframe('features', df)
    dm.save_dataframe('features', df)
    dm.save_metadata_value('key', 1)
    dm.save_missing_values(missing_values)
    dm.save_object('obj', obj)

    dm.save_images({'img1': img, 'img2':img})
    dm.save_image('img1', img)

    # gets:
    # RE-Initialize data manager:
    dm = DataManager(config={}, _input=dm.data_output_manager, _output=dm.data_output_manager)
    print('dm.get_features()')

    print(dm.get_features())

    print("dm.get_analysis_value('key')")
    print(dm.get_metadata_value('key'))

    print("dm.get_missing_values()")
    print(dm.get_missing_values())

    print("dm.get_object('obj')")
    print(dm.get_object('obj'))

    print("dm.get_all_objects()")
    print(dm.get_all_objects())

    print('''
    ###########################################
    #          model manager tests            #
    ###########################################
    ''')

    # Initialize model manager:
    mm = ModelManager(config={}, _input=dm.data_output_manager, _output=IOManager())

    # saves:
    mm.add_predictions(index_train, train_y_predict, y_train, prediction_type='TRAIN')
    mm.save_dataframe(df_name, df, to_table=None)
    mm.save_metadata_value('key', 1)
    mm.save_scaler({'a': 1})
    mm.save_object('my_obj', obj)
    mm.save_object('my_pyspark_object', spark_obj, obj_type='pyspark')
    mm.save_object('my_tensorflow_object', obj, obj_type='tensorflow')
    mm.save_object('my_keras_object', obj, obj_type='keras')
    mm.save_object('my_pytorch_object', obj, obj_type='pytorch')
    mm.save_images({'img1': img, 'img2': img})
    mm.save_image('img1', img)
    mm.save_best_model(lr_model)
    mm.save_all_models({'model1': lr_model, 'model2': lr_model})
    mm.save_best_models_scores(scores)
    mm.save_best_model_name(best_model_name)
    mm.save_best_feature_selection(best_feature_selection)
    mm.save_best_model_feature_names(selected_features)
    mm.save_best_model_feature_coefficients(coefficients)
    mm.save_metadata(analysis_results_dict)
    # mm.save_to_analysis_results(scores, selected_features, coefficients, feature_selection_method, constant=None)
    mm.save_dataframe('features', df)

    # gets:
    # RE-Initialize model manager:
    mm = ModelManager(config={}, _input=mm.model_output_manager, _output=mm.model_output_manager)
    print('mm.get_all_metadata()')
    print(mm.get_all_metadata())
    print('mm.get_features()')
    print(mm.get_features())
    print("mm.get_object('obj')")
    print(mm.get_object('obj'))
    print('mm._get_all_objects()')
    print(mm._get_all_objects())
    print("mm.get_analysis_value('key')")
    print(mm.get_metadata_value('key'))
    print('=======================================================================================================')
    print("all IOManager tests passed")
    '''
