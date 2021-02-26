# Spark Regression

This asset is implementing an regression model in spark handling continuous predicted variables.


Good examples for usage in this boilerplate are house price predictions, 
product sales, driving time, and any other quantity/continuous value.

The boilerplate is using diabetes data set to predict the fasting 
blood sugar level of pre-diagnosed diabetes patients.
Features are blood measures and other personal characteristics.


The Spark Regression asset is an implementation of the [Advanced Regression](/model-boilerplates/advanced_regression) boilerplate using the [pyspark](https://spark.apache.org/docs/latest/api/python/index.html) library.

All the data manipulations, data transformation, modelling preparation, feature selection and model training are all done in **pyspark**, using the MLApp's [feature utilities](/api/utils.features.spark) built for spark and using MLApp's [AutoML](/api/utils.automl).