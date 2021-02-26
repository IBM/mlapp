# Introduction

MLApp comes with out-of-the-box examples of assets. You can view the available boilerplates with a simple CLI command:

```bash
mlapp boilerplates show
```

## 1. Boilerplate Installation

Installing a boilerplate can be easily done with the ML App CLI:
```text
Usage: mlapp boilerplates install [OPTIONS] NAME

  Usage:

  `mlapp boilerplates install BOILERPLATE_NAME` - install mlapp
  boilerplate named BOILERPLATE_NAME

  `mlapp boilerplates install BOILERPLATE_NAME -f` - forcing boilerplate
  installation named BOILERPLATE_NAME (can override an existing boilerplate).

  `mlapp boilerplates install BOILERPLATE_NAME -r NEW_NAME` - install
  mlapp boilerplate named BOILERPLATE_NAME and rename it to NEW_NAME.

Options:
  -f, --force          Flag force will override existing asset_name file.
  -r, --new-name TEXT  Use it to rename an asset name on installation.
  -h, --help           Show this message and exit.
```

Example:

```bash
mlapp boilerplates install advanced_regression
```

This will create the [Advanced Regression](/model-boilerplates/advanced_regression) boilerplates in your `assets` directory of your MLApp project.

## 2. Available Boilerplates

- [Basic Regression](/model-boilerplates/basic_regression)
- [Advanced Regression](/model-boilerplates/advanced_regression)
- [Classification](/model-boilerplates/classification)
- [Spark Regression](/model-boilerplates/spark_regression)
- [Spark Classification](/model-boilerplates/spark_classification)
- [Flow Regression](/boilerplates/flow_regression)