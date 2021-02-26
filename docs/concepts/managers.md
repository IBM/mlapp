# Data Manager & Model Manager

The Data Manager and Model Manager are the key endpoints for your asset to the MLApp Framework.

## 1. Data Manager

The Data Manager is responsible for all data processing, including data load, clean and transform. For example, you might use this file to load data from a csv file, remove columns with nulls and perform some log transformations. In a more advanced case, you might load data from a database and perform many more transformations. The output of this file is your final dataframe that is ready to be run through feature selection and modelling algorithms.

## 2. Model Manager

The Model Manager is responsible for the modelling jobs. It takes the data returned from the data manager and runs it through the modeling process. Here is where you will be using your favorite Python algorithms to train a model, or recalling existing trained models to run a forecast. Here is also where you will be configuring and saving your model metadata, images and data required to assessing model quality.