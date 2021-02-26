# Advanced Regression

This asset is implementing an advanced regression model handling continuous predicted variables.


Good examples for usage in this boilerplate are house price predictions, 
product sales, driving time, and any other quantity/continuous value.

The boilerplate is using diabetes data set to predict the fasting 
blood sugar level of pre-diagnosed diabetes patients.
Features are blood measures and other personal characteristics.

## 1. Advanced Regression Data Manager

The **_AdvancedRegressionDataManager_** class demonstrates how to **load** the data 
from various sources and **clean** it using configurable instructions 
(e.g, which percent of NULLs will cause removal of a feature). 
The cleaning also demonstrate that one should save features' mean 
in order to fill in missing values in forecast tasks. 
**Transform** the data shows how one can create different transformation 
and control them via config file - interactions, binning, mathematical transformation
(e.g. log, exp, etc.) and finally drop out some of the created features.

This class also demonstrates how one can save time by calling the same 
inner functionalities both in train and forecast data handling tasks.

## 2. Advanced Regression Data Manager

The **_AdvancedRegressionModelManager_** class builds a model and also implements 
the prediction task given a trained model.

**train_model** shows how one can use pre built functionality of our framework 
to save development time, for example, splitting the data to X and y, 
to train and test, use MLApp's AutoML functionality that selects the best model, 
the best hyper params and best feature selection. 
It also shows how one can specify the different options for models 
(here one can specify whether to use linear models like LinearRegression, 
Ridge, Lasso etc. or non linear models like different trees based models - 
RandomForest, XGBoost etc) , with their hyper params to be searched,
 and the different options for feature selections. See more in the MLApp's [AutoML](/api/utils.automl) documentation.
 
Once the best model is selected, the data scientist is shown how to store it and 
store any relevant metadata on the selected model, then use it in the **forecast** tasks.

Once a model is trained, all its outputs is given a unique run id. 

It is worth going through the different configuration files to see the difference jobs that can be done using the asset.
   