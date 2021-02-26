# MLApp - Documentation
Welcome to IBM Services Framework for ML Applications Documentation!

## Getting Started
Detailed installation instructions can be found [here](/getting-started).

## Crash Course
A "Hello World" type tutorial - [MLApp's Crash Course](/crash-course/introduction).

## API Reference
See the [API Reference](/api/utils.automl) for more details. 




## What is MLApp?

MLapp is a Python 3 application framework for building machine learning applications that are robust and production-ready. It was developed by IBMers based on learnings from dozens of machine learning projects for IBM clients.

Data scientists today often work largely "from scratch" when embarking on a new project - they open up a notebook or [IDE](https://www.codecademy.com/articles/what-is-an-ide), import their favorite packages and start coding. This approach is OK for quick prototyping, but causes significant problems later when it is time to industrialize the work and get it deployed in a production environment.

MLapp aims to solve this key pain point. Data scientists using MLApp can build machine learning services that are configuration-based and scalable. MLApp handles machine learning job pipelines like model train and model forecast under-the-hood, allowing data scientists to focus on what they do best. MLApp also comes with [useful utilities](/api/utils.automl) including database adapters and auto-ML functionality. Finally, the framework is built in pure Python and so is easily extendable, allowing teams to build on each others work over time.

## What kind of use cases can be built in MLApp?

Anything that can be built in Python can be built in MLapp - so just about anything! Although the focus of MLApp thus far has been classical machine learning use cases, the development team is working on extensions to support advanced categories like Spark, Deep Learning and Optimization.

## What is RAD-ML? How is it related to MLApp?

RAD-ML stands for Rapid Asset Development for Machine Learning. It is a methodology and set of accelerators for building scalable ML applications.

MLApp is one of the accelerators within the larger RAD-ML method. For example, MLCP (Machine Learning Control Panel) is another accelerator, used for building machine learning user interfaces, advanced business UIs and APIs.

## Why is all of this important?

RAD-ML and MLApp were developed based on the following key learnings:

* Clients today are demanding machine learning solutions that are highly customized to their specific business needs
* At the same time, clients no longer tolerate science experiments that need to be rebuilt from the ground up in order to be deployed in production.

In this way, RAD-ML is a differentiator for IBM by providing a proven methodology that allows IBMers and client teams to deliver custom solutions quickly, that are immediately deployable in production without significant refactoring.
