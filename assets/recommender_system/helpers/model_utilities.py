import datetime, calendar, math, random, time, numbers, json, traceback
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter, OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, MeanShift, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, BayesianRidge, ElasticNet, Lars, LassoLars
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, IsolationForest,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, f1_score, make_scorer,roc_auc_score
from custom_classes import GenerativeMixture
from feature_selection import *
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import distance
from numpy import random, diag, matrix
import itertools as iter
from sklearn.metrics import silhouette_score, calinski_harabasz_score
#import xgboost as xgb
from copy import deepcopy


'''
#######  SPLIT TRAIN TEST FUNCTIONS #########
'''

def split_train_test_set_with_date(raw_X, y, split_date):
    X = raw_X.copy()
    X_train = X[(X.index <= split_date)]
    y_train = y[(y.index <= split_date)]
    X_test = X[(X.index > split_date)]
    y_test = y[(y.index > split_date)]
    return X_train.astype(float), y_train.astype(float), X_test.astype(float), y_test.astype(float)


def split_train_test_with_prop(raw_X, y, prop=0.8):
    X = raw_X.copy()
    indexes = np.arange(0, len(X))
    index_train = indexes[0: int(prop * len(X.index))]
    index_test = indexes[int(prop * len(X.index)):]
    X_train = X.copy().iloc[index_train]
    y_train = y.copy().iloc[index_train]
    X_test = X.copy().iloc[index_test]
    y_test = y.copy().iloc[index_test]
    return X_train, y_train, X_test, y_test, index_train, index_test


def split_train_test_with_prop_random(raw_X, y, prop=0.8):
    X = raw_X.copy()
    index_random = np.arange(0, len(X))
    random.shuffle(index_random)
    index_train = index_random[0: int(prop * len(index_random))]
    index_test = index_random[int(prop * len(index_random)): len(index_random)]
    X_train = X.copy().iloc[index_train]
    y_train = y.copy().iloc[index_train]
    X_test = X.copy().iloc[index_test]
    y_test = y.copy().iloc[index_test]
    return X_train, y_train, X_test, y_test, index_train, index_test


def split_train_test_with_prop_indices(raw_X, y, prop=0.8):
    X = raw_X.copy()
    index_split = int(prop * len(X.index))
    X_train = X.iloc[:index_split, :]
    y_train = y.iloc[:index_split, :]
    X_test = X.iloc[index_split:, :]
    y_test = y.iloc[index_split:, :]
    return X_train, y_train, X_test, y_test, X.index.values[0: index_split], X.index.values[index_split: len(X)]


def split_X_and_y(data, variable_to_predict):
    """
    transform the data into X,y
    :param data: full data in DataFrame
    :param variable_to_predict: variable to predict
    :return X: DataFrame with data of X
    :return y_df: DataFrame with data of y
    """
    X = data.copy()
    X = X.drop(labels={variable_to_predict}, axis=1)
    y_df = data[[variable_to_predict]]
    y_df = y_df.rename(columns={variable_to_predict: 'y'})
    return X, y_df


def rolling_cross_validation(X, train=0.8, test=0.1, cv_not_to_end=False):
    '''
    Creating cross validation split for time series
    :param X: dataframe to split
    :param train: train proportion
    :param test: test proportion
    :param cv_not_to_end: False will take the test from end of train to the end of the time series
    :return: split to train and test
    '''
    if train + test <= 1:
        index_split = int(train * len(X.index))
        test_index_end = int(test * len(X.index))
        X_train = X.iloc[:index_split, :]
        # cv reaches up to percentage defined in test variable
        if cv_not_to_end:
            X_test = X.iloc[index_split:index_split + test_index_end, :]
        # cv reaches up to the end of the series
        else:
            X_test = X.iloc[index_split:, :]
        return X_train.index.values, X_test.index.values
    else:
        return None


def split_model_respecting_prop(X, y, prop):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1-prop), random_state=0)
    try:
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    except ValueError:
        try:
            y_clusters = classify_with_dev(y, jump=1, min_samples=int(len(y)/100))
            for train_index, test_index in sss.split(X, y_clusters):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        except Exception:
            X_train, y_train, X_test, y_test, train_index, test_index = split_train_test_with_prop_random(X, y, prop)
    return X_train, y_train, X_test, y_test, train_index, test_index


'''
#######  DATA SCIENCE MODELS - AUTO MODEL SELECTION  #########
'''

def classification_models(X_train, y_train, X_test, y_test, models=None, fixed_params=None, hyperparams=None, cv=None, cv_weights=None, keys=None, scoring=None, outliers=[-2, 2], best_model_full_fit=False, low_score_is_better=False, n_jobs=1):
    #todo: knearestneighbors suffer from n_jobs=-1 , set it to n_jobs=1
    '''
    Runs through several calssification algorithm and return the best model with best hyper-parameter
    :param X_train: dataFrame with train features
    :param y_train: array of labels to predict on train
    :param X_test: dataFrame with test features
    :param y_test: array of labels to predict on test
    :param models: dictionary of models available in the functionality, see example dict in the function itself
    :param fixed_params: dictionary, initialize each model with these fixed params
    :param hyperparams: dictionary for Grid Search
    :param cv: cross validation splits
    :param cv_weights: array, weight for each split
    :param keys:  list of models to actually from the list of all models
    :param scoring: scorer for the GridSearchcv
    :param outliers: array of length(y_train) with indicators of outliers
    :param best_model_full_fit: boolean , whether to fit the best model
    with all train and test data after it was chosen
    :param low_score_is_better: boolean, i.e. for MAPE, low is better
    :return: train prediction, test prediction, best model object, intercept, coefficients, mean CV score
    '''
    #TODO: Add documentation about n+jobs in case you use XGBoost n_jobs should be equal to 1
    hyper_params_search = {}
    y_len = len(y_train)

    outliers_fraction = np.sum([np.sum(y_train == outlier) for outlier in outliers]) / float(y_len)
    if models is None:
        models = {
            'Logistic': LogisticRegression,     # binary
            'IsolationForest': IsolationForest,    # binary
            'RBF SVM': SVC,     # binary
            'MultiLogistic': LogisticRegression,
            'ExtraTreeClassifier': ExtraTreesClassifier,
            'RandomForest': RandomForestClassifier,
            'GaussianNB':GaussianNB,
            'GaussianProcess_Classifier':GaussianProcessClassifier,
            'KnearestNeighbors':KNeighborsClassifier,
            'XGBoostClassifier':xgb.XGBClassifier
        }

    if fixed_params is None:
        fixed_params = {
            'Logistic': {
                'solver':'liblinear',
                'max_iter':500
            },
            'IsolationForest': {},
            'RBF SVM': {'kernel':'rbf', 'probability': True},
            'MultiLogistic': {},
            "RandomForest": {},
            'XGBoostClassifier': {
                # 'objective': 'binary:logistic','silent': True, 'min_child_weight': 11,
                # 'sample_type': 'uniform', 'subsample': 0.8, 'normalize_type': 'tree', 'rate_drop': 0.1,
                # 'skip_drop': 0.5, 'colsample_bytree': 0.7, 'missing': 0, 'booster': 'dart', 'seed': 1337
            },
            'GaussianNB': {},
            'KnearestNeighbors': {},
            'ExtraTreeClassifier': {
                'min_samples_leaf': 4,
                'max_depth': 10,
                'class_weight': 'balanced'

            },
            'GaussianProcess_Classifier': {}
        }
    if hyperparams is None:
        hyperparams = {
            'Logistic': {
                'C': np.logspace(-4, 4, 15),
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced', None],
                'fit_intercept': [True],
            },
            "RandomForest": {"max_depth": [5, 7, 10], "n_estimators": [400, 700, 1000]},
            'MultiLogistic':{
                'penalty': ['l2'],
                'C': np.logspace(-4, 4, 15),
                'class_weight': ['balanced', None],
                'fit_intercept': [True],
                'solver': ['newton-cg', 'lbfgs'],
                'multi_class': ['multinomial'],
            },
            'XGBoostClassifier': {
                'max_depth': [5,10],
                'learning_rate': np.linspace(0.001, 0.02, 3),
                'n_estimators': [400, 700],
                'min_child_weight': [3, 10],
            },
            'IsolationForest': {
                'max_samples': [y_len],
                'contamination': [outliers_fraction],
                'random_state': [np.random.RandomState(42)],
                'n_estimators': [700]
            },
            'RBF SVM':{
                'gamma':['auto', 0.5, 2, 4]
            },
            'KnearestNeighbors': {
                'n_neighbors': [1, 2, 3, 5],
                'p': [1,2,3],
                'leaf_size': [30, 60],
                'algorithm': ['ball_tree', 'kd_tree', 'brute']
            },
            'ExtraTreeClassifier': {
                'n_estimators': [10, 50, 200],
                'min_samples_split': [2, 10],
                # 'min_samples_leaf': [4], # TODO: add more options - takes more time
                # 'max_depth': [10], # TODO: add more options - takes more time
            },
        }

    if keys is not None:
        models = {m: models[m] for m in models if m in keys}
    else:
        keys = models.keys()

    # run randomized search
    # n_iter_search = 50
    scoring = make_scorer(f1_score, average='micro') if 'IsolationForest' in keys else scoring
    if len(models) > 0:
        for key in models:
            model = models[key]
            tuned_parameters = hyperparams[key]
            if len(tuned_parameters.keys()) == 0:
                hps = GridSearchCV(model(**fixed_params[key]), tuned_parameters, n_jobs=n_jobs, cv=cv, verbose=1, refit=True, scoring=scoring)
            else:
                hps = RandomizedSearchCV(model(**fixed_params[key]), tuned_parameters, n_jobs=n_jobs, cv=cv, verbose=1, refit=True, scoring=scoring)
            start = time.time()
            hps.fit(X_train, np.ravel(y_train))
            hyper_params_search[key] = hps
            print("Hyper parameters search took %.2f seconds for model: %s"
                  " parameter settings." % ((time.time() - start), key))

        results_summary_df, best_model, intercept, coefficients = score_summary(keys=hyper_params_search.keys(),
                                                                                grid_searches=hyper_params_search,
                                                                                features=X_train.columns,
                                                                                low_score_is_better=False,
                                                                                cv_weights=cv_weights)

        train_predict = best_model.predict(X_train).reshape(1, len(X_train))[0]
        test_predict = best_model.predict(X_test).reshape(1, len(X_test))[0]
        if best_model_full_fit:
            best_model.fit(pd.concat([X_train, X_test]), np.concatenate((y_train, y_test)))

        return train_predict, test_predict, best_model, intercept, coefficients, np.abs(results_summary_df.iloc[0]['mean_score'])
    else:
        return [], [], None, 0, [], 0


def clustering_models(X_train, y_train, X_test, y_test, models=None, fixed_params=None, hyperparams=None, cv=None, cv_weights=None, keys=None, scoring=None, best_model_full_fit=False, low_score_is_better=False, n_jobs=1):
    '''
    Runs through several clustering algorithm in two manners:
     if labels exist -
     it then returns the best model with best hyper-parameter
     if labels do not exists - performs classical clustering excercise
     and returns several cluster quality measurements like sillouette , PQD etc.
    :param X_train: dataFrame with train features
    :param y_train: array of labels to predict on train
    :param X_test: dataFrame with test features
    :param y_test: array of labels to predict on test
    :param models: dictionary of models available in the functionality, see example dict in the function itself
    :param fixed_params: dictionary, initialize each model with these fixed params
    :param hyperparams: dictionary for Grid Search
    :param cv: cross validation splits
    :param cv_weights: array, weight for each split
    :param keys:  list of models to actually from the list of all models
    :param scoring: scorer for the GridSearchcv
    :param best_model_full_fit: boolean , whether to fit the best model
    with all train and test data after it was chosen
    :param low_score_is_better: boolean, i.e. for MAPE, low is better
    :return: train prediction, test prediction, best model object, intercept, coefficients, mean CV score
    '''


    grid_search = {}
    n_cluster_option = [] if y_train is None else [len(np.unique(y_train))]

    if models is None:
        models = {
            'AgglomerativeClustering': AgglomerativeClustering,
            'kmeans': KMeans,
            'MiniBatchKmeans': MiniBatchKMeans,
            'DB-SCAN':DBSCAN,
            'MeanShift':MeanShift,
            'SpectralClustering':SpectralClustering
        }

    if fixed_params is None:
        fixed_params = {
            'AgglomerativeClustering': {},
            'kmeans': {},
            'MiniBatchKmeans': {},
            'DB-SCAN':{},
            'MeanShift':{},
            'SpectralClustering':{}

        }
    if hyperparams is None:
        hyperparams = {
            'AgglomerativeClustering': {
                    'linkage': ["ward", "complete", "average"],
                    'n_clusters': [3, 8] + n_cluster_option,
            },
            'kmeans': {
                'max_iter': [100, 300, 500],
                'n_clusters': [3, 8] + n_cluster_option,
                'algorithm' : ['auto','full','elkan']
            },
            'MiniBatchKmeans': {
                'max_iter': [100, 500],
                'n_clusters': [3, 8] + n_cluster_option,
            },
            'DB-SCAN': {
                'leaf_size': [30,60]
            },
            'MeanShift': {},
            'SpectralClustering': {
                'n_clusters': [3, 8] + n_cluster_option,
            }
        }

    if keys is not None:
        models = {m: models[m] for m in models if m in keys}

    #all classes together in one model:
    if len(models) > 0:
        for model in models:
            model_instance = models[model]

            # in case y_train is none
            if y_train is not None:
                model_instance.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])

            try:
                gs = GridSearchCV(model_instance(**fixed_params[model]), hyperparams[model], n_jobs=n_jobs, cv=cv, verbose=1, refit=True, scoring=scoring)

                start = time.time()
                gs.fit(X_train, y_train)
                grid_search[model] = gs
                print("GridSearchCV took %.2f seconds for model: %s"
                      " parameter settings." % ((time.time() - start), model))
            except Exception as e:
                print(e)
                print("Warning! Model: '" + model + "' does not support 'predict' function. "
                                                    "Make your own scorer to solve this issue")

        if len(grid_search.keys()) > 0:
            results_summary_df, best_model, intercept, coefficients = score_summary(keys=grid_search.keys(),
                                                                                    grid_searches=grid_search,
                                                                                    features=X_train.columns,
                                                                                    low_score_is_better=False,
                                                                                    cv_weights=cv_weights)
        else:
            return [], [], None, 0, []
        train_predict = best_model.predict(X_train)
        test_predict = best_model.predict(X_test)
        if best_model_full_fit:
            best_model.fit(pd.concat([X_train, X_test]), np.concatenate((y_train, y_test)))

        return train_predict, test_predict, best_model, intercept, coefficients, np.abs(results_summary_df.iloc[0]['mean_score'])
    else:
        return [], [], None, 0, []


def linear_models(X_train, y_train, X_test, y_test, models=None, fixed_params=None, hyperparams=None, cv=None, cv_weights=None, keys=None, scoring=None, best_model_full_fit=False, low_score_is_better=False, n_jobs=1):
    '''
    Runs through several linear regression algorithm and return the best model with best hyper-parameter
    :param X_train: dataFrame with train features
    :param y_train: array of values to predict on train
    :param X_test: dataFrame with test features
    :param y_test: array of values to predict on test
    :param models: dictionary of models available in the functionality, see example dict in the function itself
    :param fixed_params: dictionary, initialize each model with these fixed params
    :param hyperparams: dictionary for Grid Search
    :param cv: cross validation splits
    :param cv_weights: array, weight for each split
    :param keys:  list of models to actually from the list of all models
    :param scoring: scorer for the GridSearchcv
    :param best_model_full_fit: boolean , whether to fit the best model
    with all train and test data after it was chosen
    :param low_score_is_better: boolean, i.e. for MAPE, low is better
    :return: train prediction, test prediction, best model object, intercept, coefficients, mean CV score
    '''
    hyper_params_searches = {}
    if models is None:
        models = {
            'Lasso': Lasso,
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'ElasticNet': ElasticNet,
            'Lars': Lars,
            'LassoLars': LassoLars,
            'BayesianRidge': BayesianRidge
        }

    if fixed_params is None:
        fixed_params = {
            'Ridge': {},
            'Lasso': {},
            'BayesianRidge': {},
            'LinearRegression': {},
            'ElasticNet': {},
            'Lars': {},
            'LassoLars': {}
        }

    if hyperparams is None:
        hyperparams = {
            'Ridge': {'alpha': np.logspace(-10, 10, 50)},
            'Lasso': {'alpha': np.logspace(-10, 10, 50)},
            'BayesianRidge': {'alpha_1': np.logspace(-10, 10, 5), 'alpha_2': np.logspace(-10, 10, 5),
                              'lambda_1': np.logspace(-10, 10, 5), 'lambda_2': np.logspace(-10, 10, 5)},
            'LinearRegression': {},
            'ElasticNet': {'alpha': np.logspace(-1, 5, 20), 'l1_ratio': np.geomspace(0.01, 1, 5)},
            'Lars': {'normalize': [False], 'n_nonzero_coefs': [10]},
            'LassoLars': {'alpha': np.logspace(-10, 10, 50), 'normalize': [False]}
        }

    if keys is not None:
        models = {m: models[m] for m in models if m in keys}

    if len(models) > 0:
        for key in models:
            try:
                model = models[key]
                tuned_parameters = hyperparams[key]
                start = time.time()
                if len(tuned_parameters.keys()) == 0:
                    hps = GridSearchCV(model(), tuned_parameters, cv=cv, verbose=1, refit=True, scoring=scoring, n_jobs=n_jobs)
                else:
                    hps = RandomizedSearchCV(model(), tuned_parameters, cv=cv, verbose=1, refit=True, scoring=scoring, n_jobs=n_jobs)
                hps.fit(X_train, y_train)
                hyper_params_searches[key] = hps
                print("Hyper parameters search took %.2f seconds for model: %s"
                      " parameter settings." % ((time.time() - start), key))
            except Exception as e:
                print("error: ",e)
                traceback.print_exc()
        keys = hyper_params_searches.keys()

        if len(keys) > 0:
            results_summary_df, best_model, intercept, coefficients = score_summary(keys=keys,
                                                                                    grid_searches=hyper_params_searches,
                                                                                    low_score_is_better=low_score_is_better,
                                                                                    features=X_train.columns,
                                                                                    cv_weights=cv_weights)

            train_predict = best_model.predict(X_train).reshape(1,len(X_train))[0]
            test_predict = best_model.predict(X_test).reshape(1,len(X_test))[0]
            if best_model_full_fit:
                best_model.fit(pd.concat([X_train,X_test]), np.concatenate((y_train, y_test)))
            return train_predict, test_predict, best_model, intercept, coefficients, np.abs(results_summary_df.iloc[0]['mean_score'])
        else:
            return [], [], None, 0, [], 0
    else:
        return [], [], None, 0, [], 0


def non_linear_models(X_train, y_train, X_test, y_test, models=None, fixed_params=None, hyperparams=None, cv=None, cv_weights=None, keys=None, scoring=None, best_model_full_fit=False, low_score_is_better=False, n_jobs=1):
    '''
    Runs through several non-linear regression algorithm and return the best model with best hyper-parameter
    :param X_train: dataFrame with train features
    :param y_train: array of values to predict on train
    :param X_test: dataFrame with test features
    :param y_test: array of values to predict on test
    :param models: dictionary of models available in the functionality, see example dict in the function itself
    :param fixed_params: dictionary, initialize each model with these fixed params
    :param hyperparams: dictionary for Grid Search
    :param cv: cross validation splits
    :param cv_weights: array, weight for each split
    :param keys:  list of models to actually from the list of all models
    :param scoring: scorer for the GridSearchcv
    :param best_model_full_fit: boolean , whether to fit the best model
    with all train and test data after it was chosen
    :param low_score_is_better: boolean, i.e. for MAPE, low is better
    :return: train prediction, test prediction, best model object, intercept, coefficients, mean CV score
    '''
    grid_searches = {}
    if len(X_train.columns) <= 1:
        return [], [], None, 0, [], None

    if models is None:
        models = {
            'RandomForest': RandomForestRegressor,
            'ExtraTree': ExtraTreesRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'XGBoost':xgb.XGBRegressor
        }

    if fixed_params is None:
        fixed_params = {
            'RandomForest': {'n_estimators': 400},
            'ExtraTree': {},
            'GradientBoosting': {},
            'XGBoost': {}
        }

    if hyperparams is None:
        hyperparams = {
            "RandomForest": {"max_depth": [3, 5, 7], "n_estimators": [200, 500]},

            'GradientBoosting': {
                'max_depth': [5,10],
                'learning_rate': np.linspace(0.001, 0.02, 3),
                'n_estimators': [400, 700],
                'min_child_weight': [3, 10],
            },

            'ExtraTree': {
                'n_estimators': [ 50],
                'max_depth': [3, 5, 8],
                # 'min_samples_split': [2, 10],
                # 'min_samples_leaf': [4], # TODO: add more options - takes more time
                # 'max_depth': [10], # TODO: add more options - takes more time
            },

            'XGBoost': {
                'max_depth': [3,5,8],
                'learning_rate': np.linspace(0.001, 0.02, 3),
                'n_estimators': [200,500],
                # 'min_child_weight': [3,5],
                # 'gamma': [0, 0.01, 1],
                # 'reg_alpha': [0, 0.1, 1],
                # 'reg_lambda': [0, 0.1, 1]
            },
        }

    if keys is not None:
        models = {m: models[m] for m in models if m in keys}

    if len(models) > 0:
        for model in models:
            instance = models[model]
            start = time.time()
            gs = GridSearchCV(instance(**fixed_params[model]), hyperparams[model], n_jobs=n_jobs, cv=cv, verbose=1, refit=True, scoring=scoring)  # shuffle=True, n_folds=5,
            gs.fit(X_train, np.ravel(y_train))
            grid_searches[model] = gs
            print("GridSearchCV took %.2f seconds for model: %s"
                  " parameter settings." % ((time.time() - start), model))

        if len(grid_searches.keys()) > 0:
            results_summary_df, best_model, intercept, coefficients = score_summary(keys=grid_searches.keys(),
                                                                                    grid_searches=grid_searches,
                                                                                    low_score_is_better=low_score_is_better,
                                                                                    features=X_train.columns,
                                                                                    cv_weights=cv_weights)
            train_predict = best_model.predict(X_train)
            test_predict = best_model.predict(X_test)
            if best_model_full_fit:
                best_model.fit(pd.concat([X_train,X_test]), np.concatenate((y_train, y_test)).ravel())
            return train_predict, test_predict, best_model, intercept, coefficients, np.abs(results_summary_df.iloc[0]['mean_score'])
        else:
            return [], [], None, 0, [], 0
    else:
        return [], [], None, 0, [], 0


def mixture_models(X_train, y_train, X_test, y_test, models=None, fixed_params=None, hyperparams=None, cv=None, cv_weights=None, keys=None, scoring=None, best_model_full_fit=False, low_score_is_better=False, n_jobs=1):
    '''
     Runs through Gaussian mixture clustering algorithm in two manners:
      if labels exist -
      it then returns the best model with best hyper-parameter
      if labels do not exists - performs classical clustering excercise
      and returns several cluster quality measurements like sillouette , PQD etc.
     :param X_train: dataFrame with train features
     :param y_train: array of labels to predict on train
     :param X_test: dataFrame with test features
     :param y_test: array of labels to predict on test
     :param models: dictionary of models available in the functionality, see example dict in the function itself
     :param fixed_params: dictionary, initialize each model with these fixed params
     :param hyperparams: dictionary for Grid Search
     :param cv: cross validation splits
     :param cv_weights: array, weight for each split
     :param keys:  list of models to actually from the list of all models
     :param scoring: scorer for the GridSearchcv
     :param best_model_full_fit: boolean , whether to fit the best model
     with all train and test data after it was chosen
     :param low_score_is_better: boolean, i.e. for MAPE, low is better
     :return: train prediction, test prediction, best model object, intercept, coefficients, mean CV score
     '''
    grid_search = {}
    if models is None:
        models = {
            'GaussianMixture': GenerativeMixture,
            'BayesGaussianMixture': GenerativeMixture,
        }

    if fixed_params is None:
        fixed_params = {
            'BayesGaussianMixture': {},
            'GaussianMixture': {},
        }
    if hyperparams is None:
        hyperparams = {
            'BayesGaussianMixture': {
                'density_estimator': [
                    BayesianGaussianMixture(covariance_type=cov_type, max_iter=m_iter)
                    for cov_type in ['spherical', 'diag', 'tied', 'full']
                    for m_iter in [100, 500]
                ]
            },
            'GaussianMixture': {
                'density_estimator': [
                    GaussianMixture(covariance_type=cov_type, max_iter=m_iter)
                    for cov_type in ['spherical', 'diag', 'tied', 'full']
                    for m_iter in [100, 500]
                ]
            }
        }

    if keys is not None:
        models = {m: models[m] for m in models if m in keys}

    #all classes together in one model:
    if len(models) > 0:
        for model in models:
            model_instance = models[model]
            gs = GridSearchCV(model_instance(**fixed_params[model]), hyperparams[model], n_jobs=n_jobs, cv=cv, verbose=1, refit=True, scoring=scoring)
            start = time.time()
            gs.fit(X_train, y_train)
            grid_search[model] = gs
            print("GridSearchCV took %.2f seconds for model: %s"
                  " parameter settings." % ((time.time() - start), model))
        if len(grid_search.keys()) > 0:
            results_summary_df, best_model, intercept, coefficients = score_summary(keys=grid_search.keys(),
                                                                                    grid_searches=grid_search,
                                                                                    features=X_train.columns,
                                                                                    low_score_is_better=False,
                                                                                    cv_weights=cv_weights)

            train_predict = best_model.predict(X_train)
            test_predict = best_model.predict(X_test)
            if best_model_full_fit:
                best_model.fit(pd.concat([X_train, X_test]), np.concatenate((y_train, y_test)))

            return train_predict, test_predict, best_model, intercept, coefficients, np.abs(results_summary_df.iloc[0]['mean_score'])
        else:
            return [], [], None, 0, []
    else:
        return [], [], None, 0, []



'''
#######  MODEL ACCURACIES DICTIONARIES #########
'''

def get_model_accuracy_for_regression(train_y_predict, train_y_real, test_y_predict, test_y_real):
    return {'Training Accuracy': measure_precision_with_pred(train_y_predict, train_y_real)['R^2'],
            'Testing Accuracy': measure_precision_with_pred(test_y_predict, test_y_real)['R^2'],
            'Training MAPE': get_mean_average_percentage_error(train_y_predict, train_y_real),
            'Testing MAPE':get_mean_average_percentage_error(test_y_predict, test_y_real)}


def get_model_accuracy_for_classification(train_y_predict, train_y_real, test_y_predict, test_y_real,train_y_proba=None,test_y_proba=None):

    try:
        if test_y_proba is not None and train_y_proba is not None:
            training_auc = float(roc_auc_score(train_y_real, train_y_proba))
            testing_auc = float(roc_auc_score(test_y_real, test_y_proba))
        else:
            training_auc = float(roc_auc_score(train_y_real, train_y_predict))
            testing_auc = float(roc_auc_score(test_y_real, test_y_predict))
    except:
        print('No AUC Accuracy')
        training_auc = None
        testing_auc = None

    try:
        training_accuracy = float(accuracy_score(train_y_real, train_y_predict))
        testing_accuracy = float(accuracy_score(test_y_real, test_y_predict))
    except:
        print('No Accuracy')
        training_accuracy = None
        testing_accuracy = None

    try:
        f1_score_train = f1_score_unbalanced_data(train_y_real, train_y_predict, return_value=['precision', 'recall', 'F_beta'])
        f1_score_test = f1_score_unbalanced_data(test_y_real, test_y_predict, return_value=['precision', 'recall', 'F_beta'])
    except Exception as e:
        print(e)
        print('No F1-scoring for unbalanced')
        f1_score_train = {}
        f1_score_test = {}

    return {'Jaccard Score Training': training_accuracy,
            'Jaccard Score Testing': testing_accuracy,
            'AUC Score Training': training_auc,
            'AUC Score Testing': testing_auc,
            'F_Beta Training Accuracy': f1_score_train.get('F_beta'),
            'F_Beta Testing Accuracy': f1_score_test.get('F_beta'),
            'recall Training Accuracy': f1_score_train.get('recall'),
            'recall Testing Accuracy': f1_score_test.get('recall'),
            'precision Training Accuracy': f1_score_train.get('precision'),
            'precision Testing Accuracy': f1_score_test.get('precision')}


def get_model_accuracy_and_residuals(train_y_predict, train_y_real, test_y_predict, test_y_real):
    return {'Training Accuracy': measure_precision_with_pred(train_y_predict, train_y_real.values)['R^2'],
            'Testing Accuracy': measure_precision_with_pred(test_y_predict, test_y_real.values)['R^2'],
            'Training MAPE': get_mean_average_percentage_error(train_y_predict, train_y_real.values),
            'Testing MAPE': get_mean_average_percentage_error(test_y_predict, test_y_real.values),
            'Training MASE': get_mean_average_scaled_error(train_y_predict, train_y_real),
            'Testing MASE': get_mean_average_scaled_error(test_y_predict, test_y_real)}, \
           (train_y_real - train_y_predict), (test_y_real - test_y_predict)


def get_model_accuracy_for_timeseries(train_y_predict, train_y_real, test_y_predict, test_y_real):
    return {'Training Accuracy': measure_precision_with_pred(train_y_predict, train_y_real)['R^2'],
            'Testing Accuracy': measure_precision_with_pred(test_y_predict, test_y_real)['R^2'],
            'Training MAPE': get_mean_average_percentage_error(train_y_predict, train_y_real),
            'Testing MAPE': get_mean_average_percentage_error(test_y_predict, test_y_real),
            'Training MASE': get_mean_average_scaled_error(train_y_predict, train_y_real),
            'Testing MASE': get_mean_average_scaled_error(test_y_predict, test_y_real),
            'Testing Peaks Error': get_absolute_error_for_peaks(test_y_predict, test_y_real)}


def get_absolute_error_for_peaks(y_predictions, y_actuals, std=1):
    # TODO: fix the function to fit the new "convert_value_to_std" function!

    peak_errors = {}

    # find peaks
    y_by_std = convert_value_to_std(y_actuals, np.mean(y_actuals), np.std(y_actuals))
    peak_dates = np.where(np.power(y_by_std, 2) >= std)[0]
    for peak_date in peak_dates:
        if y_actuals[peak_date] != 0:
            peak_errors[str(peak_date)] = (abs(float(y_actuals[peak_date] - y_predictions[peak_date]) / y_actuals[peak_date]))
        else:
            peak_errors[str(peak_date)] = abs(float(y_actuals[peak_date] - y_predictions[peak_date]))
    return peak_errors

def f1_score_unbalanced_data(y_true, y_pred, beta=0.5, negative_label=0, return_value='F_beta'):
    '''
    F1 Scoring function to finding anomalies in the data.
    Anomalies are any of the labels besides the "negative" label. The negative is the most prevalent case
    :param y_true: true labels
    :param y_pred: estimated labels
    :param beta: proportion of the precision in the f_beta score
    :param negative_label: the label of the most prevalent class
    :param return_value: 'F_beta', 'precision', or 'recall'
    :return: F_beta score of accuracy and recall of the anomalous classes
    '''

    y_pred_reshaped = y_pred.reshape(len(y_pred), 1)
    if isinstance(y_true,pd.DataFrame):
        y_true = y_true.values.ravel()

    # indices zero/non-zero
    non_zero_indices = np.where(y_true != negative_label)
    zero_indices = np.where(y_true == negative_label)

    # predictions
    positive_predictions = y_true[non_zero_indices] - y_pred_reshaped[non_zero_indices]
    negative_predictions = y_true[zero_indices] - y_pred_reshaped[zero_indices]

    # tp/tn/fp/fn
    TP = len(np.where(positive_predictions == 0)[0])
    TN = len(np.where(negative_predictions == 0)[0])
    FP = len(np.where(negative_predictions != 0)[0])
    FN = len(np.where(positive_predictions != 0)[0])

    # recall and precision
    outputs = {
        'recall': float(TP)/(TP+FN) if (TP+FN) != 0 else 1.0,
        'precision': float(TP)/(TP+FP) if (TP+FP) != 0 else 1.0
    }

    # F beta score
    if outputs['precision'] == 0 and outputs['recall'] == 0:
        outputs['F_beta'] = 0
    else:
        outputs['F_beta'] = (1+np.square(beta)) * outputs['precision'] * outputs['recall'] / (np.square(beta) * outputs['precision'] + outputs['recall'])
        # F_beta = (1 + np.square(beta)) * precision * recall / (np.square(beta) * precision + recall)
    if isinstance(return_value, str):
        if return_value not in outputs:
            raise Exception("Unknown return value: " + return_value)
        return outputs[return_value]

    elif isinstance(return_value, list):
        return_output = {}
        for value in return_value:
            if value not in outputs:
                raise Exception("Unknown value in 'return_value' list!")
            else:
                return_output[value] = outputs[value]
        return return_output

    else:
        raise Exception("Unknown type for 'return_value'!")



'''
#######  MODEL SELECTION FUNCTIONS  #########
'''

def select_best_regression_model(analysis_results, to_exp, measure='Testing MAPE'):
    """
    We will select best model based on R_sq Testing Accuracy and returns the train and test, and the model
    :param analysis_results: dictionary with Testing accuracy ( R_sq ) of each model in the dict
    :param to_exp:  whether y is log-transform of original y
    :return: train, test, accuracy dictionary, model object, best model name
    """
    if measure=='Testing Accuracy':
        best_model_key = max(analysis_results,key=(lambda key:analysis_results[key]['accuracy'][measure]))
    if measure=='Testing MAPE' or measure=='Testing MASE' or measure=='CV SCORE':
        best_model_key = min(analysis_results, key=(lambda key: analysis_results[key]['accuracy'][measure]))
    print(">>>>>>>>>>>>>>>>>>>> The best method is :" + best_model_key + ">>>>>>>>>>>>>>>>>")
    train_predicted = analysis_results[best_model_key]['train_prediction']
    test_predicted = analysis_results[best_model_key]['test_prediction']
    if to_exp:
        train_predicted = np.expm1(train_predicted)
        test_predicted = np.expm1(test_predicted)
    return train_predicted,test_predicted,analysis_results[best_model_key]['accuracy'],analysis_results[best_model_key]['best_model'], best_model_key

def select_best_classification_model(analysis_results, measure='f1 Testing Accuracy'):
    """
    We will select best model based on R_sq Testing Accuracy and returns the train and test, and the model
    :param analysis_results: dictionary with Testing accuracy ( R_sq ) of each model in the dict
    :return: train, test, accuracy dictionary, model object, best model name
    """
    best_model_key = max(analysis_results, key=(lambda key: analysis_results[key]['test_accuracy'][analysis_results[key]["selected_test_method_name"]]))
    print(">>>>>>>>>>>>>>>>>>>> The best method is :" + best_model_key +">>>>>>>>>>>>>>>>>")
    train_predicted = analysis_results[best_model_key]['train_prediction']
    test_predicted = analysis_results[best_model_key]['test_prediction']

    return train_predicted,test_predicted,analysis_results[best_model_key]['test_accuracy'],\
           analysis_results[best_model_key]['best_model'], best_model_key

def stack_regressors(regression_results, X_train, X_test, y_train, y_test, cv, cv_weights):
    '''
    :param regression_results:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param cv:
    :param cv_weights:
    :return:
    '''
    # Models trying for stacking
    linear_model, non_linear_model = LinearRegression(), xgb.XGBRegressor()

    # Iterating over cross validation splits and saving scores
    cv_linear_scores, cv_non_linear_scores = [], []
    for cv_index in cv:
        # new X train and X test
        regressors_train_df, regressors_test_df = pd.DataFrame(), pd.DataFrame()

        # copy model + fit and predict on cv
        for model in regression_results.keys():
            if (~np.isnan(regression_results[model]['train_prediction']).any() & ~np.isnan(regression_results[model]['test_prediction']).any()):
                model_copy = deepcopy(regression_results[model]['model'][0])
                model_copy.fit(X_train.iloc[cv_index[0]][regression_results[model]['selected_features_names']], y_train['y'][cv_index[0]])
                regressors_train_df[model] = model_copy.predict(X_train.iloc[cv_index[0]][regression_results[model]['selected_features_names']])
                regressors_test_df[model] = model_copy.predict(X_train.iloc[cv_index[1]][regression_results[model]['selected_features_names']])

        # Non-Linear
        non_linear_model.fit(regressors_train_df, y_train['y'][cv_index[0]].values)
        non_linear_accuracy = get_model_accuracy_for_timeseries(non_linear_model.predict(regressors_train_df),y_train['y'][cv_index[0]].values,
                                                                non_linear_model.predict(regressors_test_df),y_train['y'][cv_index[1]].values)
        # Linear
        linear_model.fit(regressors_train_df,y_train['y'][cv_index[0]].values)
        linear_accuracy = get_model_accuracy_for_timeseries(linear_model.predict(regressors_train_df), y_train['y'][cv_index[0]].values,
                                                            linear_model.predict(regressors_test_df), y_train['y'][cv_index[1]].values)

        cv_linear_scores.append(linear_accuracy['Testing MASE'])
        cv_non_linear_scores.append(non_linear_accuracy['Testing MASE'])

    # Calculating mean (weighted) scores
    linear_mean_mase, non_linear_mean_mase = np.sum(cv_weights * cv_linear_scores), np.sum(cv_weights * cv_non_linear_scores)

    # Preparing  full data fitted models to create X input for training stack model for full data
    regressors_train_df, regressors_test_df, regressors_full_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for model in regression_results.keys():
        regressors_train_df[model] = regression_results[model]['train_prediction']
        regressors_test_df[model] = regression_results[model]['test_prediction']
        X_full = pd.concat([X_train[regression_results[model]['selected_features_names']], X_test[regression_results[model]['selected_features_names']]])
        regressors_full_df[model] = regression_results[model]['model'][0].predict(X_full).ravel()

    # Choosing best stack model
    if linear_mean_mase < non_linear_mean_mase:
        final_model = linear_model
        best_mase = linear_mean_mase
    else:
        final_model = non_linear_model
        best_mase = non_linear_mean_mase

    # Getting estimates for train and test
    final_model.fit(regressors_train_df, y_train['y'].values)
    final_train, final_test = final_model.predict(regressors_train_df), final_model.predict(regressors_test_df)

    # Retrain on full X
    final_model.fit(regressors_full_df, pd.concat([y_train['y'], y_test['y']]))

    # Calculating Accuracy
    accuracy = get_model_accuracy_for_timeseries(final_train, y_train['y'], final_test, y_test['y'])
    accuracy['CV SCORE'] = ((5.0/6) * best_mase) + ((1.0/6) * accuracy['Testing MASE'])
    # plot_series(final_train, y_train['y'])
    # plot_series(final_test, y_test['y'])
    models_to_store = [regression_results[key]['model'] for key in regressors_test_df.columns]
    models_keys = [key for key in regressors_test_df.columns]
    underlying_coefficient = [] #TODO: fix this: get_feature_importance(X_train, regressors_train_df, final_model, regression_results)
    models_importance = [1]
    selected_feature_names = {}

    # Calculating Coefficients of all models (estimators + stack)
    try:
        underlying_coefficient = [{key: regression_results[key]['coefficients']} for key in regression_results.keys()]
        try:
            underlying_coefficient.append({models_keys[index]: final_model.coef_[index] for index, feature in enumerate(final_model.coef_)})
        except Exception as e:
            print(e)
            try:
                underlying_coefficient.append({models_keys[index]: final_model.feature_importances_[index] for index, feature in enumerate(final_model.feature_importances_)})
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
    model_names=['stacked_regression']
    model_names.extend(regression_results.keys())
    regression_results['stacked_reg'] = {
        'accuracy': accuracy,
        'train_prediction': final_train,
        'test_prediction': final_test,
        'model': [final_model],
        'selected_features_names': list(X_train.columns),
        'coefficients': underlying_coefficient,
        'models_importance': models_importance,
        'models_names': model_names
    }
    return regression_results, underlying_coefficient, models_keys, models_to_store,models_importance

def stack_classifiers(classification_results, X_train, X_test, y_train, y_test, cv, cv_weights):
    #TODO: complete this function
    pass

'''
#######  GRID SEARCH REPORTS  #########
'''

def report(grid_scores, n_top=3):
    '''
    Utility function to report best scores from a GridSearchCV object
    :param grid_scores:
    :param n_top:
    :return:
    '''
    best_models_params = []
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        best_models_params.append(score.parameters)
    return best_models_params[0]


def row(key, scores, params,cv_weights):
    '''
    Utility function for summarizing the scores of the GridSearch CV analysis
    '''
    if cv_weights is None:
        n = len(scores)
        cv_weights = np.ones(n) * 1.0 / n
    d = {'estimator': key, 'min_score': min(scores), 'max_score': max(scores),'weighted_mean_score':np.sum(cv_weights*scores) , 'mean_score': np.mean(scores), 'std_score': np.std(scores)}
    return pd.Series(dict(list(params.items()) + list(d.items())))


def score_summary(keys, grid_searches, features, sort_by='weighted_mean_score', low_score_is_better=False, cv_weights=None):
    '''
    Function to get the best model in the GridSearchCV analysis. format the output of the best model
    :param keys: all GS keys
    :param grid_searches: actual results of the GS
    :param features: coefficients of the model
    :param sort_by: CV score to sort by ( mean, min, max etc.)
    :param low_score_is_better: boolean - whether low score is better
    :param cv_weights: the weights of the CV splits
    :return: results report, the best model
    '''

    def calc_params_df(est, cv_weights):
        # get splits
        splits = list(filter(lambda x: True if all(word in x for word in ['split', 'test_score']) else False,grid_searches[est].cv_results_.keys()))
        # get hyper params names
        hyper_params_attr = list(filter(lambda x: True if all(word in x for word in ['param_']) else False,grid_searches[est].cv_results_.keys()))
        splits_df = pd.DataFrame(np.transpose([grid_searches[est].cv_results_[x] for x in splits]), columns=splits)
        output_df = pd.DataFrame()
        for column_name in hyper_params_attr:
            output_df[column_name] = grid_searches[est].cv_results_[column_name]
        output_df['max_score'] = splits_df.apply(lambda x: max(x), axis=1)
        output_df['min_score'] = splits_df.apply(lambda x: min(x), axis=1)
        output_df['mean_score'] = splits_df.apply(lambda x: np.mean(x), axis=1)
        output_df['std_score'] = splits_df.apply(lambda x: np.std(x), axis=1)
        output_df['estimator'] = est
        if cv_weights is None:
            cv_weights = 1
        output_df['weighted_mean_score'] = splits_df.apply(lambda x: np.mean(np.multiply(x, cv_weights)), axis=1)
        return  output_df

    df = pd.concat([calc_params_df(est=key,cv_weights=cv_weights) for key in keys], sort=False).sort_values(by=sort_by, ascending=low_score_is_better)
    columns = ['estimator', 'min_score', 'mean_score', 'weighted_mean_score', 'max_score', 'std_score']
    columns = columns + [c for c in df.columns if c not in columns]
    best_model = grid_searches[df.reset_index()['estimator'][0]].best_estimator_
    try:
        intercept = best_model.intercept_
        if len(best_model.coef_) == 1:
            coefficients = {features[index]: best_model.coef_[0][index] for index, feature in enumerate(best_model.coef_[0])}
        else:
            coefficients = {features[index]: best_model.coef_[index] for index, feature in enumerate(best_model.coef_)}
    except:
        try:
            intercept = 0
            coefficients = {feature: best_model.feature_importances_[index] for index, feature in enumerate(features)}
        except:
            try:
                intercept=0
                coefficients = best_model.booster().get_fscore()
            except:
                intercept = 0
                coefficients = []
    return df[columns], best_model, intercept, coefficients


def print_time(msg, last_time):
    now = time.time()
    print("> " + msg + ":", now - last_time, "seconds")
    return now


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return iter.chain.from_iterable(iter.combinations(s, r) for r in range(len(s)+1))


'''
#######  ASSISTING FUNCTIONS  #########
'''

def convert_value_to_std(series, mean, std, class_labels=[-2, -1, 0, 1, 2], classification_thresholds=[-2, -1, 1, 2]):
    '''
    Utility function to map the values of the series to standard deviation labels according to the thresholds.
    number of thresholds should be 1 less than number of labels
    :param series: the series to convert.
    :param mean: float , the mean of the series
    :param std: float, the standard deviation
    :param class_labels: the labels to map values to
    :param classification_thresholds: the thresholds fitting for the labels
    :return: labeling of the series
    '''

    def foo(x,mean,std,thresholds,classes):
        for i in range(len(classes)):
            if x>=(mean+std*thresholds[i]) and x<=(mean+std*thresholds[i+1]):
                return classes[i]
    if isinstance(series, pd.DataFrame):
        series = np.asarray(series[series.columns[0]])
    else:
        series = np.asarray(series)

    #deep copy of the input
    thresholds = list(map((lambda x: x), classification_thresholds))
    classes = list(map((lambda x: x), class_labels))

    if not all(isinstance(n, numbers.Number) for n in thresholds):
        raise Exception("Cannot unclassify Y using non numeric thresholds")

    if len(thresholds)+1 != len(classes):
        raise Exception("Cannot classify Y to unmatching classes to thresholds. "
                        "number of class labels should be = number of thresholds +1 ")
    if all(np.diff(thresholds)<=0):#descending order thresholds
            thresholds.reverse()
            classes.reverse()
    else:
        if all(np.diff(thresholds) >= 0):  # ascending order thresholds
            pass
        else:
            raise Exception("Cannot classify Y using unordered thresholds of the standard deviations. please provide sorted list of classification thresholds")
    thresholds.insert(0, -np.inf)
    thresholds.append(np.inf)

    res = list(map(lambda x: foo(x,mean,std, thresholds,classes) ,series))
    return res


def convert_std_to_value_mapper(mean, std,class_labels=[-2, -1, 0, 1, 2], classification_thresholds=[-2, -1, 1, 2]):
    thresholds = list(map((lambda x: x), classification_thresholds))
    classes = list(map((lambda x: x), class_labels))

    if not all(isinstance(n, numbers.Number) for n in thresholds):
        raise Exception("Cannot unclassify Y using non numeric thresholds")

    if all(np.diff(thresholds) <= 0):  # descending order thresholds
        thresholds.reverse()
        classes.reverse()
    else:
        if not all(np.diff(thresholds) >= 0):  # ascending order thresholds
            raise Exception(
                "Cannot classify Y using unordered thresholds of the standard deviations. "
                "Please provide sorted list of classification thresholds")

    mapper = {}
    for i in np.arange(len(thresholds)):
        mapper[thresholds[i]]=mean + thresholds[i]* std
    if 0 not in mapper:
        mapper[0] = mean
    return mapper


def classify_y_values_by_std(y_train, y_test, class_labels=[-2,-1,0,1,2], classification_thresholds=[-2,-1,1,2]):
    '''
    Map the values of the y_train and y_test to standard deviation labels according to the thresholds.
    number of thresholds should be 1 less than number of labels
    :param y_train: the series to convert, will be taken the averages and std for the mapping
    :param y_test:
    :param class_labels: the labels to map values to
    :param classification_thresholds: the thresholds fitting for the labels
    :return: labeling of the y_train, y_test
    '''
    mean_y = np.mean(y_train)
    std_y = np.std(y_train)
    y_train_new = convert_value_to_std(y_train, mean_y, std_y,class_labels,classification_thresholds)
    y_test_new = convert_value_to_std(y_test, mean_y, std_y,class_labels,classification_thresholds)
    return y_train_new, y_test_new, convert_std_to_value_mapper(mean_y, std_y,class_labels,classification_thresholds)


def score_demand_peaks_prediction(y_true, y_pred, std=1, peak_dates=None, peaks_weight=0.0):
    if peak_dates is None:
        # find peaks
        y_by_std = convert_value_to_std(y_true, np.mean(y_true), np.std(y_true))
        peak_dates = np.where(np.power(y_by_std, 2) >= std)[0]

    # if no peaks
    accuracy_peaks = 0.0
    accuracy = get_mean_average_scaled_error(y_pred, y_true)
    if len(peak_dates) > 0:
        accuracy_peaks = get_mean_average_scaled_error(y_pred, y_true, peak_dates=peak_dates)

    total_accuracy = peaks_weight * accuracy_peaks + (1.0 - peaks_weight) * accuracy
    return total_accuracy


def interpolate_missing_features_values(df, features_names, predicted_name):
    '''
    Reverse regression of the predicted variable with the feature with missing values
    :param df: the features dataFrame
    :param features_names: features to fill missing values
    :param predicted_name: the predicted variable in the dataframe
    :return: dataframe without missing values
    '''
    # returns df without nulls
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df type should be pandas.DataFrame')
    # if not (isinstance(features_names, list) or isinstance(features_names, iter)):
    #     raise TypeError('features_names type should be list')
    if not isinstance(predicted_name, str) and not isinstance(predicted_name, list):
        raise TypeError('predicted_name type should be str or list')
    result = df.copy()

    if isinstance(predicted_name, list):
        for predicted_name_i in predicted_name:
            for current_feature in features_names:
                cur_df = df[[current_feature, predicted_name_i]]
                cur_df = cur_df.dropna()
                regr_model = LinearRegression()
                regr_model.fit(cur_df[[predicted_name_i]], cur_df[[current_feature]])
                predicted_values = regr_model.predict(df[[predicted_name_i]])
                predicted_values = predicted_values.ravel()
                result[current_feature].fillna(pd.Series(predicted_values), inplace=True)
        return result
    else:
        for current_feature in features_names:
            cur_df = df[[current_feature, predicted_name]]
            cur_df = cur_df.dropna()
            regr_model = LinearRegression()
            regr_model.fit(cur_df[[predicted_name]], cur_df[[current_feature]])
            predicted_values = regr_model.predict(df[[predicted_name]])
            predicted_values = predicted_values.ravel()
            result[current_feature].fillna(pd.Series(predicted_values), inplace=True)
        return result



def drop_zero_coefficients(matrix, coefficients):
    '''
    # function get matrix dataframe and coefficients list
    # and filtering all the zero coefficients from list and zero columns from matrix
    :param matrix:
    :param coefficients:
    :return:
    '''
    columns = matrix.columns
    non_zero_columns = []
    non_zero_coefficients = []
    for i in range(len(columns)):
        if coefficients[i] != 0:
            non_zero_coefficients.append(coefficients[i])
            non_zero_columns.append(columns[i])
    return matrix[non_zero_columns], non_zero_coefficients


def scale_train_set(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    features = X_train.columns
    try:
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.values), columns=features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
    except Exception as e:
        print("Cannot Scale X features due to the following:" + e)
        raise e
    return scaler, X_train_scaled, X_test_scaled


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def classify_with_dev(y, jump, min_samples=3):
    sorted_y = y.sort_values(ascending=True)
    diff = (sorted_y - sorted_y.shift(1)) / sorted_y.shift(1)
    diff_average = np.mean(diff)

    # We keep the values that the jump between 2 consecutive values is higher that the average diff times the jump
    high_diff_index = [index for index in range(len(diff)) if diff[index] > (1 + jump) * diff_average]
    consecutives_index = [False] + [high_diff_index[i] in [high_diff_index[i - 1] + j for j in range(1, min_samples)] for i in range(1, len(high_diff_index))]
    high_diff_index = [high_diff_index[i] for i in range(len(high_diff_index)) if consecutives_index[i] == False]
    thresholds = [0] + [sorted_y.values[i - 1] for i in high_diff_index] + [max(sorted_y.values)]
    if len(thresholds) > 0:
        label_y = pd.cut(y, bins=thresholds, labels=range(1, len(thresholds)))
    else:
        label_y = [0] * len(y)

    # We eliminate extreme classes if the number of samples is lower than the min_samples.
    if len([val for val in label_y.values if val == min(label_y.values)]) < min_samples:
        label_y = label_y.replace(min(label_y.values), min(label_y.values) + 1)
        for label in set(label_y.values):
            label_y = label_y.replace(label, label - 1)

    if len([val for val in label_y.values if val == max(label_y.values)]) < min_samples:
        label_y = label_y.replace(max(label_y.values), max(label_y.values) - 1)
    return label_y



def measure_precision_with_pred(y_predictions, y_actuals):
    '''
    R square calculation and all its components
    :param y_predictions:
    :param y_actuals:
    :return:
    '''

    if isinstance(y_actuals,pd.DataFrame):
        y_actuals = y_actuals.values

    mean_y_true = np.mean(y_actuals)
    ss_total = sum([math.pow(y_actuals[index] - mean_y_true, 2) for index in range(len(y_actuals))])
    ss_reg = sum([math.pow(y_predictions[index] - mean_y_true, 2) for index in range(len(y_predictions))])
    ss_res = sum([math.pow(y_actuals[index] - y_predictions[index], 2) for index in range(len(y_actuals))])
    if ss_total > 0:
        r_squared = 1 - float(ss_res) / ss_total
    else:
        if ss_res == 0:
            r_squared = 1
        else:
            r_squared = 0
    return {'R^2': r_squared, 'SS_Tot': ss_total, 'SS_res': ss_res, 'SS_reg': ss_reg}


def get_mean_average_percentage_error(y_predictions, y_actuals):
    '''
    MAPE calculations
    :param y_predictions:
    :param y_actuals:
    :return:
    '''
    error_mean = 0
    counter = 0

    if isinstance(y_actuals, pd.DataFrame):
        y_actuals = y_actuals.values

    for i in range(len(y_actuals)):
        if y_actuals[i] != 0:
            error_mean += abs(float(y_actuals[i] - y_predictions[i]) / y_actuals[i])
        else:
            error_mean += abs(float(y_actuals[i] - y_predictions[i]) / (y_actuals[i] + 1))
        counter += 1
    if counter > 0:
        error_mean = float(error_mean) / counter
    else:
        error_mean = 0
    return float(100) * error_mean


def get_mean_average_scaled_error(y_predictions, y_actuals, peak_dates=None):
    '''
    MASE calculations
    :param y_predictions:
    :param y_actuals:
    :param peak_dates:
    :return:
    '''
    y_a = y_actuals.copy()
    y_p = y_predictions.copy()
    if peak_dates is None:
        naive_model_error = np.mean(np.abs(y_a[1:] - np.roll(y_a, 1)[1:]))
    else:
        # remove zero index if exists because can't "naive" predict it
        peak_dates_no_zero_index = peak_dates.copy()
        zero_index = np.argwhere(peak_dates == 0)
        if len(zero_index) > 0:
            peak_dates_no_zero_index = np.delete(peak_dates_no_zero_index, zero_index)
        naive_model_error = np.mean(np.abs(y_actuals[peak_dates_no_zero_index] - y_actuals[peak_dates_no_zero_index - 1]))
        y_a = y_actuals[peak_dates_no_zero_index]
        y_p = y_predictions[peak_dates_no_zero_index]

    # MASE
    if naive_model_error != 0:
        mase = np.mean(np.abs(y_a-y_p)/naive_model_error)
    # MAD/Mean Ratio (when naive_model_error is zero)
    else:
        y_actuals_mean = np.mean(y_a)
        if y_actuals_mean == 0:
            y_actuals_mean = y_actuals_mean + 1
        mase = np.mean(np.abs(y_a - y_p) / y_actuals_mean)
    return mase

def plot_series(train_data, y_data):
    """
    Plots y against train of y
    """
    fig2, ax2 = plt.subplots()
    ax2.plot(train_data, 'b--')
    ax2.plot(y_data, 'g:')
    plt.show()


def normalized_weights(model, X, y):
    weights = [[feature, model.params[feature]] for feature in model.params.index]
    scaled_weights = []
    min_y = min(y.values)
    max_y = max(y.values)
    for feature_tuple in weights:
        feature = feature_tuple[0]
        weight = feature_tuple[1]
        values_features = X[feature].values
        min_feature = min(values_features)
        max_feature = max(values_features)
        if max_y > min_y:
            scaled_weights.append([feature, weight * (max_feature - min_feature) / (max_y - min_y)])
        else:
            scaled_weights.append([feature, weight])
    return scaled_weights



################################################################################################################################
##########################################                                 #####################################################
##########################################    Clustering algorithm Start   #####################################################
##########################################                                 #####################################################
################################################################################################################################



def get_clusters_centers(data, labels):
    '''
    calculates clusters centers by mean
    :param data: pandas DataFrame, shape(obsevations,features)
    :param labels: 1D array
    :return: pandas DataFrame, cluster centers
    '''
    try:
        if isinstance(labels, pd.DataFrame):
            raise Exception('labels should be 1D array')

        if ~isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        labels = pd.DataFrame(labels, columns=['labels'])
        df = pd.concat([data, labels], axis=1)
        centers = pd.DataFrame.groupby(df, by='labels', as_index=True).agg('mean')
        return centers
    except Exception as e:
        print(e)
        raise e

def mean_euclidian_distance_in_cluster(Cr):
    nr = len(Cr.index)
    df1 = Cr.loc[np.repeat(Cr.index, nr)]
    df2 = Cr.loc[np.tile(Cr.index, nr)]
    return np.true_divide(1, 2 * nr) * sum(sum(np.power(np.subtract(df1.values[:, :-1], df2.values[:, :-1]), 2)))

def mean_euclidian_distance_in_cluster2(Cr):
    nr = len(Cr.index)
    if nr <= 1: return 0
    m1 = np.array([Cr.index.values for i in list(range(nr))])
    m2 = m1.T
    df1 = Cr.loc[m1[np.triu_indices_from(m1, 1)]]
    df2 = Cr.loc[m2[np.triu_indices_from(m2, 1)]]
    return np.true_divide(1, nr) * sum(sum(np.power(np.subtract(df1.values[:, :-1], df2.values[:, :-1]), 2)))


def optimal_k_gap_statistic(data, Wks, n_clusters, n_observations, refs=None, B=5, rand_estimator=None,
                            estimators_params=None, rand_solver='full'):
    '''
    Return optimal K (number of clusters) according to Gap Statistic score
    :param data: pandas Dataframe or 2d array
    :param Wks: array or list, of Wk mean distances
    :param n_clusters: array or list, of K's (number of clusters for each Wk)
    :param n_observations: int, number of data n_observations
    :param refs: (optional) None or 2D array of distribution matrix
    :param B: (optional) int, number of random iterations
    :param rand_estimator: string, cluster estimator name
    :param estimators_params: list of python Dictionaries
    :param rand_solver: string, how to calculates Wk, possible values - ['centroids','full'] default 'full'
    :return: int, K
    '''
    try:
        n_features = data.shape[1]

        def rand_estimator_centroids(rands, k):
            rands_df = pd.DataFrame(rands)
            estimator = rand_estimator(**estimators_params[k])
            estimator.fit(rands_df)
            try:
                labels_rand = estimator.predict(rands_df)
            except Exception as e:
                labels_rand = estimator.labels_

            centers_rand = get_clusters_centers(rands_df, labels_rand)
            return calc_W_by_centroids_distance(rands_df, labels_rand, centers_rand)

        def rand_estimator_all_points(rands, k):
            rands_df = pd.DataFrame(rands)
            estimator = rand_estimator(**estimators_params[k])
            estimator.fit(rands_df)
            labels_rand = estimator.predict(rands_df)
            return calc_W_by_all_points_distance(rands_df, labels_rand)

        def random_points_distribution(data, n_observations, n_features, B):
            tops = data.max(axis=0)
            bots = data.min(axis=0)
            dists = matrix(diag(tops - bots))

            rands = random.random_sample(size=(n_observations, n_features, B))
            for i in range(B):
                rands[:, :, i] = rands[:, :, i] * dists + bots
            return rands

        solver_method = {
            'centroids': rand_estimator_centroids,
            'full': rand_estimator_all_points
        }

        if ~isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if rand_estimator is None:
            rand_estimator = KMeans
        else:
            rand_estimator = eval(rand_estimator)

        if refs is None:
            rands = random_points_distribution(data.values, n_observations, n_features, B)
        else:
            rands = refs

        Wks_B = list(map(lambda k: list(map(lambda i: solver_method.get(rand_solver, 'full')(rands[:, :, i], k), range(B))),range(len(n_clusters))))
        l_tags = list(map(lambda Wk_B: np.mean(np.log(Wk_B)), Wks_B))
        SDks = [np.sqrt(np.mean(np.power(np.subtract(np.log(Wk_B), l_tags[i]), 2))) for i, Wk_B in enumerate(Wks_B)]
        s_k = np.multiply(SDks, np.sqrt(1 + np.true_divide(1, B)))
        gaps = l_tags - np.log(Wks)
        optimal_k = None
        for k in range(0, len(gaps) - 1):
            gap_k = gaps[k]
            gap_k_plus_1 = gaps[k + 1]
            if gap_k >= (gap_k_plus_1 - s_k[k + 1]):
                optimal_k = n_clusters[k]
                break
        return optimal_k

    except Exception as e:
        print(e)
        raise e


def optimal_k_krzanowski_and_lai(Wks, n_clusters, n_features):
    '''
    Return optimal K (number of clusters) according to Hartigan score
    IMPORTANT NOTE: this implementation ignores Wk both edges
    :param Wks: array or list, of Wk mean distances
    :param n_clusters: array or list, of K's (number of clusters for each Wk)
    :return: int, K
    '''

    if len(Wks) != len(n_clusters):
        raise Exception('Error: Wks length should be equal to n_clusters')

    try:
        p = n_features

        def diff(k, Wk, Wk_minus_1):
            two_div_p = np.true_divide(2, p)
            diff = (np.power((k - 1), two_div_p) * Wk_minus_1) - (np.power(k, two_div_p) * Wk)
            return diff

        it1, it2, it3 = iter.tee(Wks, 3)
        next(it2, None)
        next(it3, None)
        next(it3, None)
        max_KL = -1
        max_k = -1
        for i, (Wk_minus_1, Wk, Wk_plus_1) in enumerate(zip(it1, it2, it3)):
            k = n_clusters[i]
            if k == 1:
                continue
            KL = np.abs(np.true_divide(diff(k, Wk, Wk_minus_1), diff(k + 1, Wk_plus_1, Wk)))
            if KL > max_KL:
                max_KL = KL
                max_k = k

        if max_k == -1:
            max_k = None

        return max_k

    except Exception as e:
        print(e)
        raise e


def optimal_k_hartigan(Wks, n_clusters, n_observations, threshold=10):
    '''
    Return optimal K (number of clusters) according to Hartigan score
    IMPORTANT NOTE: this implementation ignores Wk right edge
    :param Wks: array or list, of Wk mean distances
    :param n_clusters: array or list, of K's (number of clusters for each Wk)
    :param n_observations: int, number of data n_observations
    :param threshold: int, returns K > threshold
    :return: int K ot None if none of the K's is optimal (bigger than threshold)
    '''

    if len(Wks) != len(n_clusters):
        raise Exception('Error: Wks length should be equal to n_clusters')

    try:
        it1, it2 = iter.tee(Wks, 2)
        next(it2, None)
        optimal_k = None
        for i, (Wk, Wk_plus_1) in enumerate(zip(it1, it2)):
            k = n_clusters[i]
            H = np.multiply(np.true_divide(Wk, Wk_plus_1) - 1, n_observations - k - 1)
            if H < threshold:
                optimal_k = k
                break

        return optimal_k
    except Exception as e:
        print(e)
        raise e

optimal_k_selections_methods = {
    'gap': optimal_k_gap_statistic,
    'h': optimal_k_hartigan,
    'kl': optimal_k_krzanowski_and_lai
}

def majority_agreement(decision_array, agreement_min=2):
    '''
    :param decision_array: list or pd.Series.
    with each vector represent decisions of one judge/ model on each cell.
    Currently only supports equality in judges choice not ranges
    (i.e. all agree that cell_i is higher than some value x_i)
    if only one valus to be determined among several judges/models, one should create a df with one row , multiple columns or
    an array of 1 element arrays
    :param agreement_min: the minimum of judges need to agree to make a decision
    :return: array where each cell is the majority voting for it
        if number of judges id equal or less than that minimum, return empty []
    '''

    c = Counter(decision_array)
    most_common = c.most_common()[0]

    if (most_common[1] >= agreement_min):
        return most_common[0]
    else:
        return -1

def optimal_k_selections(data, Wks, range_clusters, rand_estimator, estimators_params, algo='full',rand_solver='centroids'):
    '''
    :param data:
    :param Wks:
    :param range_clusters:
    :param rand_estimator:
    :param estimators_params:
    :param algo:
    :param rand_solver:
    :return:
    '''
    try:
        n_observations = data.shape[0]
        n_features = data.shape[1]

        algo_dict = {
            'gap': {
                'function': optimal_k_selections_methods.get('gap'),
                'params': {
                    'Wks': Wks[1:], # ignoring first index since was calculated in previous func call
                    'n_clusters': range_clusters[1:],  # ignoring first index since was calculated in previous func call
                    'n_observations': n_observations,
                    'rand_estimator': rand_estimator,
                    'estimators_params': list(estimators_params),
                    'rand_solver': rand_solver,
                    'data': data
                }
            },
            'h': {
                'function': optimal_k_selections_methods.get('h'),
                'params': {
                    'Wks': Wks[1:], # ignoring first index since was calculated in previous func call
                    'n_clusters': range_clusters[1:], # ignoring first index since was calculated in previous func call
                    'n_observations': n_observations,
                }
            },
            'kl': {
                'function': optimal_k_selections_methods.get('kl'),
                'params': {
                    'Wks': Wks,
                    'n_clusters': range_clusters,
                    'n_features': n_features,
                }
            }
        }

        if algo == 'full':
            optimal_k_krzanowski_and_lai = algo_dict['kl']['function'](**algo_dict['kl']['params'])
            optimal_k_gap_statistic = algo_dict['gap']['function'](**algo_dict['gap']['params'])
            optimal_k_hartigan = algo_dict['h']['function'](**algo_dict['h']['params'])
            ma = majority_agreement([optimal_k_krzanowski_and_lai, optimal_k_gap_statistic, optimal_k_hartigan])
            if ma == -1:
                optimal_k = optimal_k_gap_statistic
            else:
                optimal_k = ma
        else:
            optimal_k = algo_dict.get(algo, 'gap')['function'](**algo_dict.get(algo, 'gap')['params'])

        if optimal_k is None:
            return (None, -1)
        else:
            index = range_clusters.index(optimal_k)
            return (optimal_k, index)

    except Exception as e:
        print(e)
        raise e

def score_summary_level_3(X,k, estimator_name, estimators, scoring):
    '''
    Function to get the best model in the GridSearchCV analysis. format the output of the best model
    :param keys: all GS keys
    :param results: actual results of the GS
    :param features: coefficients of the model
    '''

    try:

        estimators_params = map(lambda est: est['estimators_params'],estimators)
        W_ks = list(map(lambda est: est['W_k'],estimators))
        scoring_params = {
            'data': X,
            'Wks': W_ks,
            'rand_estimator': estimator_name,
            'range_clusters': [k-1,k,k+1],
            'estimators_params': estimators_params
        }
        (optimal_k, optimal_k_index) = scoring['function'](**scoring_params)

        return True if (optimal_k is not None and optimal_k > -1) else False

    except Exception as e:
        print('Error: ', e)
        raise e

def kullback_leibler_divergence_score(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    P - is a uniform disribution vector, for example: [0.33,0.33,0.33]
    Q - is a disribution of interest vector, for example: [0.2,0.5,0.3]
    output score: can be negative and positive, optimal is zero
    """

    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence


def KL_distribution(est):
    try:
        counter_labels_values = Counter(est.labels_).values()
        Q_distribution = np.array(list(map(lambda x: np.true_divide(x, len(counter_labels_values)), counter_labels_values)))
        P_distribution = np.true_divide(np.ones(len(counter_labels_values)), len(counter_labels_values))
        result = kullback_leibler_divergence_score(P_distribution, Q_distribution)
        return result
    except Exception as e:
        print(e)
        raise e

def calc_W_by_centroids_distance(data, labels, centroids):
    '''
    calc W by the formula:
    W = sum((1/2nr)*sum((Xij - Centroid_i_tagj)^2))
    nr - number of obsevations in cluster
    i - observation
    j - feature in observation
    :param data: pandas DataFrame
    :param labels: cluster estimator labels array
    :param centroids: pandas DataFrame
    :return: W float
    '''

    try:

        labels_df = pd.DataFrame(labels, columns=['labels'])
        data_copy = data.copy()
        data_copy = pd.concat([data_copy, labels_df], axis=1)
        clusters = pd.DataFrame.groupby(data_copy, by='labels', as_index=True)
        W = sum([(sum(map(lambda r: np.power(distance.euclidean(r[:-1], centroids.loc[label]), 2), Cr.values))) for label, Cr in clusters])
        return W
    except Exception as e:
        print("Error:",e)
        raise e


def calc_W_by_all_points_distance(data, labels):
    '''
    calc W by the formula:
    W = sum((1/2nr)*sum((Xij - Xi_tagj)^2))
    nr - number of obsevations in cluster
    i - observation
    j - feature in observation
    :param data: pandas DataFrame
    :param labels: cluster estimator labels array
    :return: W float
    '''

    def calc_cluster_points_euclidian_distance(Cr):
        nr = len(Cr.index)
        if nr <= 1: return 0
        m1 = np.array([Cr.index.values for i in range(nr)])
        m2 = m1.T
        df1 = Cr.loc[m1[np.triu_indices_from(m1, 1)]]
        df2 = Cr.loc[m2[np.triu_indices_from(m2, 1)]]
        return np.true_divide(1, nr) * sum(sum(np.power(np.subtract(df1.values[:, :-1], df2.values[:, :-1]), 2)))

    try:
        labels_df = pd.DataFrame(labels, columns=['labels'])
        data_copy = data.copy()
        data_copy = pd.concat([data_copy, labels_df], axis=1)
        clusters = pd.DataFrame.groupby(data_copy, by='labels', as_index=True)
        W = sum([calc_cluster_points_euclidian_distance(Cr) for label, Cr in clusters])
        return W
    except Exception as e:
        print("Error:", e)
        raise e


def calc_W_k_mean_distance(data, labels, solver='full', log=False):
    '''
    Calculates Wk mean distance for k clusters
    Wk =
    :param data: pandas DataFrame or 2D array
    :param labels: list or array of clusters labels
    :param solver: string - 'full' = distance of all points, 'centroids' = distance of points for centroid same as full much more efficient
    :param log: boolean, if log the result
    :return: float, Wk mean distance
    '''

    solver_method = {
        'centroids': lambda data, labels: calc_W_by_centroids_distance(data, labels,get_clusters_centers(data, labels)),
        'full': lambda data, labels: calc_W_by_all_points_distance(data, labels)
    }

    try:
        if ~isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        Wk = solver_method.get(solver, 'full')(data, labels)

        if log:
            return np.log(Wk)
        else:
            return Wk

    except Exception as e:
        print('Error:',e)
        raise e

def score_summary_cluster_with_optimal_k(X, estimators, scoring):
    '''
    Function chooses best estimator
    :param X: pdandas Dataframe of np.array
    :param estimators: list of fitted estimators
    :param scoring: python Dictionary of functions
    :return: best estimator
    '''

    try:
        if len(estimators) == 1:
            return estimators[0]

        mtx = []
        scoring_weights= list()
        for scoring_key,scoring_method in scoring.items():
            scoring_results = list(map(lambda est: scoring_method['function'](est,X) if scoring_method['kwargs'].get('greater_is_better',True) else np.multiply(scoring_method['function'](est,X),-1),estimators))
            scoring_weights.append(scoring_method['kwargs'].get('weight',1))
            mtx.append(scoring_results)

        mtx = np.transpose(mtx)
        results_df = pd.DataFrame(mtx,columns=list(scoring.keys()))
        scaler = MinMaxScaler()
        scaler.fit(results_df)
        scaled_results_array = scaler.transform(results_df)
        scaled_results_df = pd.DataFrame(scaled_results_array, columns=list(scoring.keys()))
        results_df = np.multiply(scaled_results_df,scoring_weights)
        scoring_sum = list(results_df.sum(axis=1))
        max_index = scoring_sum.index(max(scoring_sum))

        # getting max estimator scores
        max_estimator_scores = {}
        i = 0
        for scoring_key in scoring:
            max_estimator_scores[scoring_key] = {}
            max_estimator_scores[scoring_key]['score'] = mtx[max_index][i]
            max_estimator_scores[scoring_key]['weight'] = scoring_weights[i]
            i+=1

        return estimators[max_index],max_index,max_estimator_scores

    except Exception as e:
        print('Error: ', e)
        raise e


def get_param_grid(fixed_params, hyper_params):
    '''
    Creates all combinations
    :param fixed_params: python dictionary
    :param hyper_params: python dictionary
    :return: List of python dictionaries
    '''
    hyper_params_ordered = OrderedDict(hyper_params)
    fixed_params_ordered = OrderedDict(fixed_params)
    hyper_keys = list(hyper_params_ordered.keys())
    fixed_keys = list(fixed_params_ordered.keys())
    combinations = iter.product(*(hyper_params_ordered[key] for key in hyper_keys))
    values = map(lambda x: list(x), combinations)
    values = map(lambda v: v + list(fixed_params_ordered.copy().values()), values)
    result = map(lambda v: dict(zip(hyper_keys + fixed_keys, v)), values)
    return result

def agglomerative_clustering_constraint(params_combination,X_train,fixed_params,hyper_params):
    # check if params_combination stands in agglomerative clustering constraint
    '''
    This method checks if params_combination stands in agglomerative clustering constraint and
    checks for connectivity parameters
    :param params_combination: array of dictionaries
    :param X_train: panda Dataframe
    :param fixed_params: dictionary
    :param hyper_params: dictionary
    :return: new params_combination
    '''

    def __inner_connectivity_combinations__():
        # creates connectivity instances from connectivity combinations
        params_to_return = []
        for connectivity_params in connectivity_params_combinations:
            # copy agglomerative params
            param = deepcopy(params_combination)

            # init connectivity estimator
            connectivity_params['X'] = X_train
            connectivity_instance = eval(connectivity_name)(**connectivity_params)

            # set agglomerative params with new connectivity
            param[connectivity] = connectivity_instance
            params_to_return.append(param)

        return params_to_return

    connectivity = 'connectivity'
    linkage = 'linkage'

    if connectivity in params_combination and params_combination[connectivity] is not None:
        # so create connectivity instance

        params_to_return = None

        # get connectivity estimator name
        connectivity_dict_value = params_combination[connectivity]
        if isinstance(connectivity_dict_value, dict):
            connectivity_name = list(connectivity_dict_value.keys())[0]
        else:
            connectivity_name = connectivity_dict_value

        # checks if connectivity is in fixed_params and hyper_params and create new params
        if connectivity in fixed_params and connectivity not in hyper_params:
            connectivity_name = list(connectivity_dict_value.keys())[0]
            connectivity_params = connectivity_dict_value[connectivity_name]
            connectivity_params['X'] = X_train
            connectivity_instance = eval(connectivity_name)(**connectivity_params)
            params_to_return = params_combination
            params_to_return[connectivity] = connectivity_instance

        elif connectivity not in fixed_params and connectivity in hyper_params:
            hyper_params_connectivity = hyper_params[connectivity][connectivity_name]
            connectivity_params_combinations = get_param_grid({},hyper_params_connectivity)
            params_to_return = __inner_connectivity_combinations__()

        elif connectivity in fixed_params and connectivity in hyper_params:
            hyper_params_connectivity = hyper_params[connectivity][connectivity_name]
            fixed_params_connectivity = fixed_params[connectivity][connectivity_name]
            connectivity_params_combinations = get_param_grid(fixed_params_connectivity, hyper_params_connectivity)
            params_to_return = __inner_connectivity_combinations__()
        else:
            # default connectivity is None
            params_combination[connectivity] = None


    return params_to_return


def check_for_clustering_constrains(all_params_combinations,model_name,X_train,fixed_params,hyper_params):
    # check for clustering models constraints and for nested hyper parameters

    new_params_combination = []

    for params_combination in all_params_combinations:

        new_param = params_combination
        if model_name == "AgglomerativeClustering":
            new_param = agglomerative_clustering_constraint(params_combination,X_train,fixed_params,hyper_params)

        # check new param type, can be list or dict
        if isinstance(new_param,list):
            new_params_combination.extend(new_param)
        else:
            new_params_combination.append(new_param)


    return new_params_combination



def clustering_models_unsupervised(X_train, num_clusters_params, models=None, fixed_params=None, hyper_params=None, scoring_params=None, keys=None):
    '''
    Runs through several clustering algorithm in two manners:
     if labels exist -
     it then returns the best model with best hyper-parameter
     if labels do not exists - performs classical clustering excercise
     and returns several cluster quality measurements like silhouette , PQD etc.
    :param X_train: dataFrame with train features
    :param num_clusters_params: python Dicationary for estimator number of clusters variable name
    :param models: dictionary of models available in the functionality, see example dict in the function itself
    :param fixed_params: dictionary, initialize each model with these fixed params
    :param hyper_params: dictionary for Grid Search
    :param keys:  list of models to actually from the list of all models
    # :param cv: cross validation splits
    # :param cv_weights: array, weight for each split
    # :param scoring: scorer for the GridSearchCV
    # :param best_model_full_fit: boolean , whether to fit the best model
    with all train and test data after it was chosen
    :return: train prediction, test prediction, best model object, intercept, coefficients, mean CV score
    '''

    print("----------- Runs Clustering Models Unsupervised ------------")
    if models is None:
        models = {
            'AgglomerativeClustering': AgglomerativeClustering,
            'KMeans': KMeans,
            'MiniBatchKmeans': MiniBatchKMeans,
            'DB-SCAN': DBSCAN,
            'MeanShift': MeanShift,
            'SpectralClustering': SpectralClustering
        }
    if fixed_params is None:
        fixed_params = {
            'AgglomerativeClustering': {},
            'KMeans': {},
            'MiniBatchKmeans': {},
            'DB-SCAN': {},
            'MeanShift': {},
            'SpectralClustering': {}

        }
    if hyper_params is None:
        hyper_params = {
            'AgglomerativeClustering': {
                'linkage': ["ward", "complete", "average"],
            },
            'KMeans': {
                'max_iter': [100, 300, 500],
                'algorithm': ['auto', 'full', 'elkan']
            },
            'MiniBatchKmeans': {
                'max_iter': [100, 500],
            },
            'DB-SCAN': {
                'leaf_size': [30, 60]
            },
            'MeanShift': {},
            'SpectralClustering': {
            }
        }
    if scoring_params is None:
        scoring_params = {
            "scoring_params_between_estimators": {  # level 2
                "silhouette_score": {
                    "function": lambda est, X: silhouette_score(X, est.labels_),
                    "kwargs": {
                        'greater_is_better': True,  # max = True, min = False
                        'weight': 0.5
                    }
                },
                "calinski_harabaz_score": {
                    "function": lambda est, X: calinski_harabaz_score(X, est.labels_),
                    "kwargs": {
                        'greater_is_better': True,  # max = True, min = False
                        'weight': 0.25
                    }
                },
                "KL_distribution": {
                    "function": lambda est, X: np.abs(KL_distribution(est)),
                    "kwargs": {
                        'greater_is_better': False,  # max = True, min = False
                        'weight': 0.25
                    }
                },
            },
            "scoring_params_for_k_clusters": {  # level 3
                "function": optimal_k_selections,
                "kwargs": {
                    'algo': 'full'
                }
            }
        }

    if keys is not None:
        models = {m: models[m] for m in models if m in keys}

    # estimators = list()
    best_estimators_level_3 = list()

    def fit_inner_estimator(k,params_combination):
        params_combination[num_clusters_params['methods'][model]] = k
        estimator = model_object(**params_combination)
        estimator.fit(X_train)

        try:
            estimator.labels_ = estimator.predict(X_train)
        except Exception as e:
            pass

        W_k = calc_W_k_mean_distance(X_train, estimator.labels_, 'centroids', log=False)
        return {'estimator': estimator, 'estimators_params': params_combination,"W_k": W_k}



    if len(models) > 0:

        ######################## Level 3 scoring ########################
        start = time.time()
        print("Starting to Train and calculate estimators level 3...........")
        for model in models:
            model_object = models[model]
            model_fixed_params = fixed_params.get(model, {})
            model_hyper_params = hyper_params.get(model, {})

            all_params_combinations = get_param_grid(model_fixed_params, model_hyper_params)
            # check for nested hyper_params and params constrains validation
            all_params_combinations = check_for_clustering_constrains(all_params_combinations, model, X_train,model_fixed_params,model_hyper_params)

            for params_combination in all_params_combinations:
                k = 2
                estimator_k_minus_1 = fit_inner_estimator(k-1,params_combination.copy())
                estimator_k = fit_inner_estimator(k,params_combination.copy())
                while True:
                    estimator_k_plus_1 = fit_inner_estimator(k+1,params_combination.copy())
                    is_optimal_k = score_summary_level_3(X_train,k,model,[estimator_k_minus_1,estimator_k,estimator_k_plus_1],scoring_params['scoring_params_for_k_clusters'])
                    if is_optimal_k:
                        print("For",model,"with params: ",params_combination, "The Optimal K is:",k)
                        estimator_k['Optimal K'] = k
                        best_estimators_level_3.append(estimator_k)
                        break
                    estimator_k_minus_1 = estimator_k
                    estimator_k = estimator_k_plus_1
                    k+=1



        print("level 3 scoring completed took %.2f seconds.", ((time.time() - start)))

        start = time.time()
        ######################## Level 2 scoring ########################
        print("Starting to calculates estimators scores level 2...........")
        best_estimators_level_3_models = list(map(lambda est_dict: est_dict['estimator'],best_estimators_level_3))
        best_estimator_level_2,best_estimator_level_2_index,best_estimator_level_2_scores  = score_summary_cluster_with_optimal_k(X_train,best_estimators_level_3_models,scoring_params['scoring_params_between_estimators'])
        print("Level 2 Best Estimator:", best_estimator_level_2)
        print("level 2 scoring completed took %.2f seconds.", ((time.time() - start)))
        best_estomator_result =  best_estimators_level_3[best_estimator_level_2_index]
        best_estomator_result ['Level_2_scores'] = best_estimator_level_2_scores
        return best_estomator_result

    else:
        return None


def run_cluster_models(X_train, clustering_models_options, clustering_methods):
    clustering_methods_dict = {
        'Clustering': {
            'function': clustering_models_unsupervised,
            'params':{
                'models': clustering_models_options.get('Clustering', {}).get('models', None),
                'num_clusters_params': clustering_models_options.get('Clustering', {}).get('num_clusters_params', None),
                'hyper_params': clustering_models_options.get('Clustering', {}).get('hyper_params', None),
                'fixed_params': clustering_models_options.get('Clustering', {}).get('fixed_params', None),
                'scoring_params': clustering_models_options.get('Clustering', {}).get('scoring_params', None),
            }
        },
        'Mixture Clustering': {
            'function': clustering_models_unsupervised,
            'params': {
                'models': clustering_models_options.get('Mixture Clustering', {}).get('models', None),
                'num_clusters_params': clustering_models_options.get('Mixture Clustering', {}).get('num_clusters_params', None),
                'hyper_params': clustering_models_options.get('Mixture Clustering', {}).get('hyper_params', None),
                'fixed_params': clustering_models_options.get('Mixture Clustering', {}).get('fixed_params', None),
                'scoring_params': clustering_models_options.get('Mixture Clustering', {}).get('scoring_params', None),
            }
        }
    }

    gen_models = (m for m in clustering_methods_dict if m in clustering_methods)
    best_clustering_methods_list = list()
    for clustering_model_key in gen_models:
        best_clustering = clustering_methods_dict[clustering_model_key]['function'](X_train,**clustering_methods_dict[clustering_model_key]['params'])
        best_clustering.update({'clustering_method':clustering_model_key})
        best_clustering_methods_list.append(best_clustering)

    return best_clustering_methods_list

def cluster_features(X_train, config={}, scoring_df=None):

    clustering_methods = config.get('clustering_methods',['Clustering'])
    dimensionality_reduction_methods = config.get('dimensionality_reduction_methods',['PCA'])

    dimensionality_reduction = config.get('dimensionality_reduction_options', {
        "run_on_all_features": False,
        "models":{
            "PCA": PCA,
            "TSNE": TSNE,
        },
        "fixed_params": {
            "PCA": {
                 "copy": True,
                 "whiten": False,
                 "svd_solver": 'full',
                 "tol": 0,
                 "iterated_power": 'auto',
                 "random_state": None,
            },
            "TSNE": {
                "perplexity":30,
                "early_exaggeration":12,
                "learning_rate":200,
                "n_iter":1000,
                "n_iter_without_progress":300,
                "min_grad_norm":1e-07,
                "metric":'euclidean',
                "init":'random',
                "verbose":0,
                "random_state":None,
                "method":'barnes_hut',
                "angle":0.5
             },
        },
        "hyper_params": {
            "PCA": {
                "n_components": [2,3,'mle']
            },
            "TSNE": {
                "n_components": [2, 3]
            }
        },
        "scoring_params": {
            "silhouette_score": {
                "function": lambda est, X: silhouette_score(X, est.labels_),
                "kwargs": {
                    'greater_is_better': True,# could be 'max or 'min'
                    'weight': 0.5
                }
            }
        }
     })

    clustering_models_options = config.get('clustering_models_options', {
        'Clustering': {
            "models":{
                "AgglomerativeClustering": AgglomerativeClustering,
                "KMeans": KMeans,
                "MiniBatchKMeans": MiniBatchKMeans,
                #'DB-SCAN':DBSCAN,
                #'MeanShift':MeanShift,
                #'SpectralClustering':SpectralClustering
            },
            "fixed_params":{
                "AgglomerativeClustering": {
                     "affinity":'euclidean',
                     "memory":None,
                     "connectivity":None,
                     "compute_full_tree":'auto',
                },
                "KMeans": {
                     "init":'k-means++',
                     "n_init":10,
                     "max_iter":300,
                     "tol":0.0001,
                     "precompute_distances":'auto',
                     "verbose":0, "random_state":None,
                     "copy_x":True,
                     "n_jobs":1,
                },
                "MiniBatchKMeans": {
                     "init":'k-means++',
                     "max_iter":300,
                     "batch_size":100,
                     "verbose":0,
                     "compute_labels":True,
                     "random_state":None,
                     "tol":0.0,
                     "max_no_improvement":10,
                     "init_size":None,
                     "n_init":3,
                     "reassignment_ratio":0.01
                },
                'DB-SCAN': {
                    "eps":0.5,
                    "min_samples":5,
                    "metric":'euclidean',
                    "metric_params":None,
                    "algorithm":'auto',
                    "leaf_size":30,
                    "p":None,
                    "n_jobs":1
                },
                'MeanShift': {
                    "bandwidth":None,
                    "seeds":None,
                    "bin_seeding":False,
                    "min_bin_freq":1,
                    "cluster_all":True,
                    "n_jobs":1
                },
                'SpectralClustering': {
                    "k":8,
                    "mode":None,
                    "random_state":None,
                    "n_init":10
                }
            },
            "hyper_params": {
                "AgglomerativeClustering": {
                    "linkage":['ward','complete'],
                },
                "KMeans": {
                    "algorithm":['auto','full']
                },
                "MiniBatchKMeans": {

                },
                'DB-SCAN':{

                },
                'MeanShift':{

                },
                'SpectralClustering':{

                }
            },
            "num_clusters_params": {
                "flag":True,
                "methods":{
                    "AgglomerativeClustering": "n_clusters",
                    "KMeans": "n_clusters",
                    "MiniBatchKMeans": "n_clusters",
                    'DB-SCAN':"n_clusters",
                    'MeanShift':"n_clusters",
                    'SpectralClustering':"n_clusters"
                }
            },
            "scoring_params": {
                "scoring_params_between_estimators": { # level 2
                    "silhouette_score": {
                        "function": lambda est, X: silhouette_score(X, est.labels_),
                        "kwargs": {
                            'greater_is_better': True,# could be 'max or 'min'
                            'zero_is_better': True,# could be 'max or 'min'
                            'weight': 0.5
                        }
                    },
                    "calinski_harabaz_score": {
                        "function": lambda est, X: calinski_harabasz_score(X, est.labels_),
                        "kwargs": {
                            'greater_is_better': True,# could be 'max or 'min'
                            'weight': 0.25
                        }
                    },
                    "KL_distribution": {
                        "function": lambda est, X: np.abs(KL_distribution(est)),
                        "kwargs": {
                            'greater_is_better': False,# could be 'max or 'min'
                            'weight': 0.25
                        }
                    },
                },
                "scoring_params_for_k_clusters": { # level 3
                    "function": optimal_k_selections,
                    "kwargs": {
                        'algo':'full'
                    }
                }
            }
        },
        'Mixture Clustering': {
            "models":{
                'GaussianMixture': GaussianMixture,
                'BayesianGaussianMixture': BayesianGaussianMixture,
            },
            "fixed_params":{
                'BayesianGaussianMixture': {
                    'covariance_type': 'full',
                    'tol': 0.001,
                    'reg_covar': 1e-06,
                    'max_iter': 100,
                    'n_init': 1,
                    'init_params': 'kmeans',
                    'weight_concentration_prior_type': 'dirichlet_process',
                    'weight_concentration_prior': None,
                    'mean_precision_prior': None,
                    'mean_prior': None,
                    'degrees_of_freedom_prior': None,
                    'covariance_prior': None,
                    'random_state': None,
                    'warm_start': False,
                    'verbose': 0,
                    'verbose_interval': 10
                },
                'GaussianMixture': {
                    'covariance_type': 'full',
                    'tol': 0.001,
                    'reg_covar': 1e-06,
                    'max_iter': 100,
                    'n_init': 1,
                    'init_params': 'kmeans',
                    'weights_init': None,
                    'means_init': None,
                    'precisions_init': None,
                    'random_state': None,
                    'warm_start': False,
                    'verbose': 0,
                    'verbose_interval': 10
                }
            },
            "hyper_params": {
                'BayesianGaussianMixture': {

                },
                'GaussianMixture': {

                }
            },
            "num_clusters_params": {
                "flag":True,
                "methods":{
                    "BayesianGaussianMixture": "n_components",
                    "GaussianMixture": "n_components"
                }
            },
            "scoring_params": {
                "scoring_params_between_estimators": { # level 2
                    "silhouette_score": {
                        "function": lambda est, X: silhouette_score(X, est.labels_),
                        "kwargs": {
                            'greater_is_better': True,# could be 'max or 'min'
                            'zero_is_better': True,# could be 'max or 'min'
                            'weight': 0.5
                        }
                    },
                    "calinski_harabaz_score": {
                        "function": lambda est, X: calinski_harabasz_score(X, est.labels_),
                        "kwargs": {
                            'greater_is_better': True,# could be 'max or 'min'
                            'weight': 0.25
                        }
                    },
                    "KL_distribution": {
                        "function": lambda est, X: np.abs(KL_distribution(est)),
                        "kwargs": {
                            'greater_is_better': False,# could be 'max or 'min'
                            'weight': 0.25
                        }
                    },
                },
                "scoring_params_for_k_clusters": { # level 3
                    "function": optimal_k_selections,
                    "kwargs": {
                        'algo':'full'
                    }
                }
            }
        }
    })
    if scoring_df is not None:
        for model_option in clustering_models_options:
            clustering_models_options[model_option]['scoring_params']['scoring_params_between_estimators']['market_share_score'] = {
                'function': lambda est, X: calc_W_k_mean_distance(scoring_df, est.labels_, solver='centroids'),
                'kwargs': {
                    'greater_is_better': False,
                    'weight': config.get('custom_scoring_weight', 0.25)
                }
            }
    svd_best_clusters = list()
    if dimensionality_reduction.get('run_on_all_features', True):
        print('All Features: total of', X_train.shape[1], 'features')
        best_clusters_methods = run_cluster_models(X_train, clustering_models_options, clustering_methods)
        map(lambda est: est.update({'svd_model': None,'svd_model_key': 'All Features'}),best_clusters_methods)
        svd_best_clusters += best_clusters_methods

    print('-------Start Testing SVD Algorithms -------')

    for svd_model_key, svd_model_function in dimensionality_reduction.get('models').items():

        if svd_model_key not in dimensionality_reduction_methods:
            continue

        model_svd_fixed_params = dimensionality_reduction.get('fixed_params')[svd_model_key]
        model_svd_hyper_params = dimensionality_reduction.get('hyper_params')[svd_model_key]
        all_combinations = get_param_grid(model_svd_fixed_params, model_svd_hyper_params)
        for combination in all_combinations:
            print(svd_model_key, 'parameters:', combination)
            svd_model = svd_model_function(**combination)
            custom_X = svd_model.fit_transform(X_train)
            best_clusters_methods = run_cluster_models(custom_X, clustering_models_options, clustering_methods)
            map(lambda est:est.update({'svd_model': svd_model, 'svd_model_key': svd_model_key + '_' + str(combination['n_components'])}),best_clusters_methods)
            svd_best_clusters += best_clusters_methods

    ########### scoring level 1 ################
    print("Starting to calculates estimators scores level 1...........")
    if len(svd_best_clusters) == 1:
        return svd_best_clusters[0]
    estimators = list(map(lambda est_dict: est_dict['estimator'], svd_best_clusters))
    best_cluster_model,best_cluster_model_index,best_cluster_model_scores = score_summary_cluster_with_optimal_k(X_train,estimators,scoring=dimensionality_reduction.get('scoring_params'))

    svd_best_estomator_result = svd_best_clusters[best_cluster_model_index]
    svd_best_estomator_result['Level_1_scores_svd'] = best_cluster_model_scores
    print("Done ...........")
    return svd_best_estomator_result


################################################################################################################################
###########################################                                #####################################################
###########################################    Clustering algorithm End    #####################################################
###########################################                                #####################################################
################################################################################################################################