# Part 5 - Classifier Ensembler

In the final part of the Advanced Course, we will build a model consisted of 2 different sub-models:

- The Crash Course asset converted to a neural network classifier which was done in [Part 4 - AutoML](/advanced-topics/part-4-automl).
- The Advanced Course asset which uses the **Classification Boilerplate** built in [Part 1 - Model Boilerplates](/advanced-topics/part-1-boilerplates).

The main idea behind this is to have 2 models trained:
 
- The Advanced Course asset based on the **Classification Boilerplate** that uses MLApp's AutoML features which includes automatic search for the best model using various feature selection and model approaches using best validation practices.
- The Crash Course asset using a neural network classifier model.

Afterwards you'll be running a final model that will ensemble these two sub-models together in order to take advantage of both sub-models and have a powerful classifier for the glass dataset.


## 1. Prerequisites - Crash Course and Advanced Course Assets

Make sure you have finished Parts 1-4 of the Advanced Course in your current project.


## 2. Create the Classification Ensembler Asset
Run the following command in the root of your project:
```bash
mlapp assets create classification_ensembler
```

## 3. Create the Flow Configuration

To run this complex model, we'll be running a Flow of pipelines and finally run a Flow-summarizer asset.
As was noted in part 2, the general structure of the flow config with a summarizer is as follows:
```json
{
    "pipelines_configs": 
    [
      
      {
          "data_settings": {
            ...
          },
          "model_settings": {
            ...
          },
          "flow_settings": {
            "input_from_predecessor": [...],
            "return_value": [...]
          },
          "job_settings": {
            ...
          }
      },
     
      {
          "model_settings": {
            ...
          },
          "flow_settings": {
            "input_from_predecessor": [...],
            "return_value": [...]
          },
          "job_settings": {
            ...
          }
      }
  ],
  
  "flow_config": {
      "data_settings": {
        ...
      },
      "model_settings": {
        ...
      },
      "job_settings": {
        ...
      }
  }
}
```

Lets fill in the missing parts:
We start with the first pipeline which is the classification boilerplate model:

**First**: Copy the above structure into the `assets > classification_ensembler > configs > classificatio_ensembler_train_config.yaml` configuration file.

**Second**: Copy the `data_settings`, `model_settings`, `job_settings` of `assets > advanced_course > configs > advanced_course_glass_train_config.yaml` into the first pipeline config

**Third**: In the `flow_setting`s fill in the following:
```json
"flow_settings": {
  "input_from_predecessor": [],
  "return_value": ["features", "model", "model_class_name"]
} 
```
Lets continue with the second pipeline which is the crash course asset:

**First**: we copy `model_settings` from `assets > crash_course > configs > crash_course_train_config.yaml` into the second pipeline config in the above array.

**Second**: In its `flow_settings` fill in the following:
```json
"flow_settings": {
  "input_from_predecessor": ["features"],
  "return_value": ["model", "model_class_name"]
}
```
**Third**: In the `job_settings` lets put the following :
```json
"job_settings": {
  "asset_name": "crash_course",
  "pipeline": ["load_features_from_predecessor","train_model"]
}
```

!!! note "Forwarding Model Object"
    
    When using the`self.save_automl_result` function, the model is being saved as an object under the key `model`.


## 4. Create the Flow Summarizer configuration

Next step would be to create the configuration of the Flow Summarizer which is a new key `flow_config` in your configuration file `assets > classification_ensembler > configs > classification_ensembler_train_config.yaml`:

```json
{
    "pipelines_configs": 
    [
      // crash course config
      { 
        "data_settings": { ... },
        "model_settings": { ... },
        "flow_settings": {
            "input_from_predecessor": [],
            "return_value": ["features", "model", "model_class_name"]
        }, 
        "job_settings": { ... } },
      // classification config
      { 
        "data_settings": { ... }, 
        "model_settings": { ... }, 
        "flow_settings": {
            "input_from_predecessor": ["features"],
            "return_value": ["model", "model_class_name"]
        }, 
        "job_settings": { ... }
       }
    ],
    "flow_config": {
        "data_settings": {
            "df_key_name": "features"
        },
        "model_settings": {
            "target": "Type",
            "train_size": 0.8
        },
        "job_settings": {
            "asset_name": "classification_ensembler",
            "pipeline": ["load_input", "train_model"]
        }
    }
}
```

!!! tip "Train Test Split Consistency" 
    
    Add argument `random_state=2` to the `train_test_split` method used in both assets `crash_course` and `advanced_course` at the **Model Managers** to keep consistency of train & test for the ensemble.

## 5. Classification Ensembler - Data Manager

Go to `assets > classification > classification_ensembler_data_manager.py` and add a new Pipeline function `load_input` function:

```python
@pipeline
def load_input(self, *args):
    # load predecessor pipelines' outputs 
    pipelines_input = args[0]

    # init variables
    models = []
    data = None

    # loop inputs
    for i in range(len(pipelines_input)):
        # load model
        models.append({
            'object': pipelines_input[i]['model'][0],
            'name': pipelines_input[i]['model_class_name'][0]
        })
        # load data
        if self.data_settings['df_key_name'] in pipelines_input[i]:
            data = pipelines_input[i][self.data_settings['df_key_name']][0]

    return {
        'data': data,
        'models': models
    }
```

!!! note "Predecessors' Outputs"
    
    The outputs from the previous Pipelines are passed in the args in a list ordered by the order of the Pipelines in your Flow.

## 6. Classification Ensembler - Model Manager

The plan here is to do a classification ensembler using Scikit-Learn's VotingClassifier.

Go to `assets > classification > classification_ensembler_model_manager.py`.

Add imports:

```python
from mlapp.utils.metrics.pandas import classification
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
``` 
 
Implement the `train_model` function:

```python
@pipeline
def train_model(self, data_input):
    data = data_input['data']
    models = data_input['models']

    # splitting X and y
    target = self.model_settings['target']
    y = data[target]
    X = data.drop(target, axis=1)

    # split the data to test and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=self.model_settings.get('train_size', 0.8), random_state=2)

    # create a dictionary of our models
    estimators = []
    for model in models:
        estimators.append((model['name'], model['object']))

    # create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='soft')

    # fit model to training data
    ensemble.fit(X_train, y_train)

    # predict
    train_predicted = ensemble.predict(X_train)
    test_predicted = ensemble.predict(X_test)

    #  calculate scores
    scores = classification(
        train_predicted, test_predicted, y_train, y_test, unbalanced=False)
    
    # print scores
    print("Ensemble Scores: ")
    for key in scores:
        print(f">> {key}: {scores[key]}")
    
    # add trained model instance
    self.save_object('ensemble_model', ensemble)

    # add model measurements
    self.save_metadata('scores', scores)
```

## 7. Run the model

Run the model via `run.py`:
```json
{
    "asset_name": "classification_ensembler",
    "config_path": "assets/classification_ensembler/configs/classification_ensembler_train_config.yaml",
    "config_name": "classification_ensembler_config"
}
```

You're done!!! you have run your first complex Flow in MLApp! Compare the results with the previous ones!