# Data Handling

Data handling within ML App should all be done by using the supported functions in the [Managers' Shared Functions](/api/managers/#shared-functions).

You have get/save functions for each type of data:

- **Data Frames**: Any Pandas/Spark data-frame object are saved to `CSV` type files.
- **Metadata**: Any static type data that you have, such as - strings, numbers, etc, in dictionary or list data structure and is `JSON` serializable. 
- **Images**: Any [matplotlib](https://matplotlib.org/) figure or [plotly](https://plotly.com/python/) figure or image.
- **Run-time Objects**: Any run-time python objects that are "pickable" (serializable/deserializable with the [pickle](https://docs.python.org/3/library/pickle.html) module).

> Note: **pyspark** objects are also supported in the run-time objects type.           

Once you use the supported functions in the managers, all the data will be saved/loaded for you locally or in any resource (db, s3 bucket, etc) automatically - once you have those resources set up in your [environment](/concepts/environment).

Once data is saved it is linked to the current run via the **run_id**.
 
## run_id

The _run_id_ is the framework's way to "uniquify" each run. Each run gets it's own _run_id_ and whatever you saved in a run can be accessed in a different run using that _run_id_.
 
In the local environment, the _run_id_ and any of it's related metadata is saved into a `CSV` file - `output_logs.csv`, in the `outputs` directory of your root project.

When plugging a database in your [environment](/concepts/environment), the _run_id_ and it's metadata is saved in the database.

The _run_id_ can also be accessed anywhere in your Data Manager/Model Manager via `self.run_id`.

## Loading Data Using the _run_id_

When you want to load data from another run just use the _run_id_ in the `job_settings` of the configuration, in example:
```json
{
  "job_settings": {
    "data_id": "<run_id>",
    "reuse_features_id": "<run_id>",
    "model_id": "<run_id>"
  }
}
```

!!! note "_data_id_, _model_id_ and _reuse_features_id_ have different purposes"

    **data_id**: used to load any data that was saved in the Data Manager.
    
    **model_id**: used to load any data that was saved in the Model Manager.
    
    **reuse_features_id**: used to load features for the Model Manager that were created using the special `feature_engineering` pipeline.



