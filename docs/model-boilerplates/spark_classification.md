# Spark Classification

This asset is implementing a classification model in spark handling binary/categorical predicted variables.


Good examples for usage in this boilerplate are churn predictions, 
health status, product levels of quality, and any other binary/categorical value.

The boilerplate is using breast cancer data set to predict whether a 
patients' breast tumor is malignant or benign.
Features are different tumor measures.


The Spark Classification asset is an implementation of the [Classification](/model-boilerplates/classification) boilerplate using the [pyspark](https://spark.apache.org/docs/latest/api/python/index.html) library.

All the data manipulations, data transformation, modelling preparation, feature selection and model training are all done in **pyspark**, using the MLApp's [feature utilities](/api/utils.features.spark) built for spark and using MLApp's [AutoML](/api/utils.automl).