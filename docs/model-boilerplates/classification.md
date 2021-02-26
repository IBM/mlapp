# Classification

This asset is implementing an advanced classification model handling binary/categorical predicted variables.


Good examples for usage in this boilerplate are churn predictions, 
health status, product levels of quality, and any other binary/categorical value.

The boilerplate is using breast cancer data set to predict whether a 
patients' breast tumor is malignant or benign.
Features are different tumor measures.

## 1. Classification Data Manager

The **_ClassificationDataManager_** class demonstrates how to **load** the data
 and **clean** it using configurable instructions 
(e.g, which percent of NULLs will cause removal of a feature). 
The cleaning also demonstrate that one should save features' mean 
in order to fill in missing values in forecast tasks. 
The class **_ClassificationFeatureEngineering_()** is showing when doing heavy 
data engineering work how to separate the different tasks and call them 
in a way to maintain readability. Take a look at thorough filling missing values method for example.
**Transform** the data shows how one can create different transformation 
and control them via config file - interactions, binning, mathematical transformation
(e.g. log, exp, etc.) and finally drop out some of the created features.
In this part we have done heavy work around handling categorical features
that oftentimes may result in exploding the feature set.
We have shown how one can evaluate the categorical features 
by checking the recall of each category level with binary predicted 
variable to decide which levels are best to keep for modeling purposes.

This class also demonstrates how one can save time by calling the same 
inner functionalities both in train and forecast data handling tasks.

## 2. Classification Model Manager
 
The **_ClassificationModelManager_** class builds a model and also implements 
the prediction task given a trained model.

**train_model** shows how one can use pre built functionality of our framework 
to save development time, for example, splitting the data to X and y, 
to train and test, use AutoML functionality that selects the best model, 
the best hyper params and best feature selection. 
It also shows how one can specify the different options for models 
(here one can specify which classifier to use - 
RandomForest, XGBoost, SVM etc.), with their hyper params to be searched,
 and the different options for feature selections. See more in the MLApp's [AutoML](/api/utils.automl) documentation.
 
Once the best model is selected, the data scientist is shown how to store it and 
store any relevant metadata on the selected model, then use it in the forecast tasks.

Once a model is trained, all its outputs is given a unique run id. 

It is worth going through the different config files to see the difference tasks can be done using the asset.

