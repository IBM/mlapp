# Introduction to the MLApp Docs

These docs help you learn to use MLApp, from your first project setup to asset boilerplates, integrations and other more advanced topics.

## What is MLApp?

MLApp is a Python library for building scalable data science solutions that meet modern software engineering standards. It was originally created as an internal framework for IBM consultants to more quickly and consistently develop production-grade machine learning code when working on projects for large enterprise clients. The project was open sourced in 2021 in an effort to share our learnings and best practices with the wider community.

Data scientists today often work largely "from scratch" when embarking on a new initiatve - they open up a notebook or [IDE](https://www.codecademy.com/articles/what-is-an-ide), import their favorite packages and start putting down code. Although this approach may work for a quick prototype or research project, it leads to significant issues later when it is time to industrialize the work and get it deployed in a production environment.

MLApp aims to solve this key pain point. Data scientists using MLApp can build machine learning services that are configuration-based and scalable. MLApp handles machine learning job pipelines like model train and model forecast under-the-hood, allowing data scientists to focus on what they do best. MLApp also comes with [useful utilities](/api/utils.automl) including database adapters and auto-ML functionality. Finally, the framework is built in pure Python and so is easily extendable, allowing teams to build on each others work over time.

## What kind of use cases can be built in MLApp?

Anything that can be built in Python can be built in MLapp - so just about anything! Although the focus of MLApp thus far has been classical machine learning use cases, the development team is working on extensions to support advanced categories like Spark, Deep Learning and Optimization.

## Why is all of this important?

MLApp is solving for the following dichotomy:

* Organizations today are demanding data science solutions that are highly customized to their specific business needs
* At the same time, organizations no longer tolerate science experiments that need to be rebuilt from the ground up in order to be deployed in production and supported over time

## Assumptions

These docs assume that you are already comfortable with [Python in the context of data science](https://www.coursera.org/learn/python-data-analysis). Additionally, you should be comfortable with common Python data science packages like Pandas, Python IDEs and git. If you aren't quite there, we recommend taking some free online courses. Here are some good ones recommended by the MLApp development team:

- [Introduction to Pandas](https://www.youtube.com/watch?v=otCriSKVV_8)
- [PyCharm video series](https://www.youtube.com/playlist?list=PLQ176FUIyIUZ1mwB-uImQE-gmkwzjNLjP), in particular [Code Navigation](https://www.youtube.com/watch?v=jmTo5xTRka8&list=PLQ176FUIyIUZ1mwB-uImQE-gmkwzjNLjP&index=6), [Debugging](https://www.youtube.com/watch?v=QJtWxm12Eo0&list=PLQ176FUIyIUZ1mwB-uImQE-gmkwzjNLjP&index=7) and [VCS topics](https://www.youtube.com/watch?v=jFnYQbUZQlA&list=PLQ176FUIyIUZ1mwB-uImQE-gmkwzjNLjP&index=11)
- [Git crash course](https://www.atlassian.com/git/tutorials)


## Feedback

We highly encourage feedback and contributions from the community!

- Submit feature suggestions or report a bug [here](https://github.com/IBM/mlapp/issues)

- Submit a pull request [here](https://github.com/IBM/mlapp/pulls). Before doing so, please refer to [our guidelines]([./CONTRIBUTING.md](https://github.com/IBM/mlapp/blob/README/CONTRIBUTING.md))