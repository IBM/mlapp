# Part 3 - Developing the Model Manager

Now open up the file **assets/crash_course/crash_course_model_manager.py**.

This file has only two functions, _train_model_ and _forecast_. These functions receive as an input the final data frame created from the data manager.

Add the following imports at the top of the file:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
```

In this section, we will implement the function _**train_model**_.

First, split your X (dataframe of features) and your y (target vector):

```python
# splitting X and y
target = self.model_settings['target']
y = data[target]
X = data.drop(target, axis=1)
```

Further splits your dataframes into train set and your test set:

```python
# split the data to test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.model_settings.get('train_percent'))
```

Train your model using the RandomForestClassifier from scikit-learn:

```python
# run classification
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

Calculate the model train and test quality - in this case, f1 score:

```python
# get model f1 score
train_estimations = model.predict(X_train)
test_estimations = model.predict(X_test)
scores = {
    'f1_score (train set)': float(f1_score(y_train, train_estimations, average='weighted')),
    'f1_score (test set)': float(f1_score(y_test, test_estimations, average='weighted'))
}
```

Use `self.save_object('model', model)` to save your trained model. MLApp automatically saves the model for later as a python [pickle](https://docs.python.org/3/library/pickle.html):

```python
# save model instance
self.save_object('model', model)
```

Finally, save the model metrics using `self.save_metadata('scores', scores)` method. This method allows you to save any JSON compatible metadata about the model for later analysis:

```python
# add model metrics
self.save_metadata('scores', scores)
```

We only added one new configuration option, so update the **crash_course_train_config.json** file accordingly:

```python
"model_settings": {
    "target": "Type",
    "train_percent": 0.8
}
```


At the end of the process, your file **assets/crash_course/crash_course_model_manager.py** should look like that:

```python
from mlapp.managers import ModelManager
from mlapp.utils import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class CrashCourseModelManager(ModelManager):
    @pipeline
    def train_model(self, data):
        # splitting X and y
        target = self.model_settings['target']
        y = data[target]
        X = data.drop(target, axis=1)

        # split the data to test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.model_settings.get('train_percent'))

        # run classification
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # get model f1 score
        train_estimations = model.predict(X_train)
        test_estimations = model.predict(X_test)
        scores = {
            'f1_score (train set)': float(f1_score(y_train, train_estimations, average='weighted')),
            'f1_score (test set)': float(f1_score(y_test, test_estimations, average='weighted'))
        }

        # save model instance
        self.save_object('model', model)

        # add model metrics
        self.save_metadata('scores', scores)

    @pipeline
    def forecast(self, data):
        raise NotImplementedError()
    
    @pipeline
    def refit(self, data):
        raise NotImplementedError()

```

And your file **assets/crash_course/configs/crash_course_train_config.json** should look like this:

```json
{
  "pipelines_configs": [
    {
      "data_settings": {
        "file_path": "data/glass.csv",
        "features_to_log": ["Na"]
      },
      "model_settings": {
          "target": "Type",
          "train_percent": 0.8
      },
      "job_settings": {
        "asset_name": "crash_course",
        "pipeline": "train"
      }
    }
  ]
}
```

<br/>
Congrats, the modelling part of the pipeline is complete! We can move on to [running the pipeline](/crash-course/part-4-running-pipeline).