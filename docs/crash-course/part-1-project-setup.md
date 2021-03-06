# Part 1 - Project Setup

## Prepare Your Workspace

Create a new empty directory for this crash course and run this command:
```bash
mlapp init
```

!!! note "Default Project Structure"

    This command will create the default project structure recommended by the MLApp's team.
    
    **assets** (Required): This folder is used to store the assets you develop.
    
    **run.py** (Required): This file is used to run your models locally.
    
    **app.py** (Required): This file is used to run your models either locally or in production.
    
    **config.py** (Required): This file contains settings for the MLApp app (should not be edited).
    
    **common** (Optional): This folder is used to create custom utilities that can be shared across your models.
    
    **data** (Optional): This folder can be used to store different data files for debugging/testing.

## Build Asset Scaffolding

Data scientists only need to work with a few files generated by the **Asset Scaffolding** to build a new asset.

Creating the Asset Scaffolding is easy using the MLApp's CLI - run this command from your root project directory: 
```bash
mlapp assets create crash_course
```

> Note: `crash_course` is the name of the asset created. We'll be using this asset name for the rest of the crash course.

!!! note "About MLApp's Asset Scaffolding"
    
     This command has created a new folder called `crash_course` inside the assets directory. It contains all the required files for developing a new model inside the ML App framework. When running a model pipeline (like model train or forecast), MLApp knows how to use these files as instructions in the pipeline process. Below is a high level summary of what each file is used for - you will be working directly in these files for the remainder of the crash course.
    
     **crash_course_data_manager.py**: The data manager is responsible for all data processing, including data load, clean and transform. For example, you might use this file to load data from a csv file, remove columns with nulls and perform some log transformations. In a more advanced case, you might load data from a database and perform many more transformations. The output of this file is your final dataframe that is ready to be run through feature selection and modeling algorithms.
    
     **crash_course_model_manager.py**: The model manager is responsible for the modeling jobs. It takes the data returned from the data manager and runs it through the modeling process. Here is where you will be using your favorite Python algorithms to train a model, or recalling existing trained models to run a forecast. Here is also where you will be configuring and saving your model metadata, images and data required to assessing model quality.
    
     **configs**: directory containing model base configurations like train and forecast.
    
     ML App enforces an important concept in production-ready machine learning that we call **configuration-based modeling**. This concept means that instead of hard coding all of their decisions about data handling and modeling, data scientists externalizing decisions to a JSON file that is injected into the code. Working this way allows for the final machine learning service to be deployed and accept jobs like train and forecast, which can now be customized by changing the configurations (i.e. job instructions) when the job is sent. For example, decisions like which features to include in the model, which transformations to apply and which model should be used (if several have been implemented) are good opportunities for configuration.
    
     **configs/crash_course_train_config.json**  and **configs/crash_course_forecast_config.json** are basic configuration files that contain all the information that is necessary for the train and forecast pipelines. The next section contains a detailed description of the config structure.  

### Configuration Files Structure

MLApp provides an easy way to inject configurations from these json objects into your data manager and model manager. You are responsible for developing a comprehensive configuration file so that future data scientists have sufficient flexibility to use the asset when it's in a deployed state. Many examples of model configurations will be provided in this crash course and in the example models provided in the framework.

Configuration files must be build using the following structure:
```javascript
{
    "pipelines_configs": [
        {
            "data_settings": {},  
            "model_settings": {},  
            "job_settings": {
                "asset_name": "...",      // e.g "crash_course"
                "pipeline": "...",        // e.g "train" or "forecast"
                "model_id": "..."	  // used in forecast pipelines
            }
        },
        {
            ...
        }
    ]
}
```

!!! note "Configuration Structure"

    * **pipelines_configs** is a list of one or more model configs. Each containing:
    * **data_settings**: settings related to the data - which are used by the data manager.    
    * **model_settings**: settings related to the model - which are used by the model manager.    
    * **job_settings**: settings related to the job in general - which are used by the job manager (not something that you will be touching much except in advance cases).  


## Get Some Data

For this simple example, we will be using the [glass dataset](https://www.kaggle.com/uciml/glass) as a source dataset.

When working in development, it is common to use flat CSV files. We will typically put files in the data folder for the development phase. In production, usually data will be loaded from a database or object storage.

For convenience, we added the data files in our repository.

#### Download The Files and Add Them to Your Project

1. Click on [data/glass.csv](https://github.com/ibm/mlapp/blob/master/data/glass.csv) & [data/glass_forecast.csv](https://github.com/ibm/mlapp/blob/master/data/glass_forecast.csv).
2. Click `Raw` to open the files on github in raw format.
3. Use your browser to save the current page (Usually by doing `File > Save Page As..`). Make sure to save it as a CSV (extension: .csv)
4. Save the files as `glass.csv` / `glass_forecast.csv` depending on the link you opened.
5. Move the files to your new project workspace inside the `data` directory.

This dataset contains 214 samples and 10 features. Each datapoint corresponds to a glass sample classified into one of the 7 categories. 10 features describe the datapoint: they correspond to some elements' measurements. 

The objective is to build a classifier for these groups.

<br/>
Now that we have our project setup ready we can move on to [developing the data manager](/crash-course/part-2-developing-the-data-manager).