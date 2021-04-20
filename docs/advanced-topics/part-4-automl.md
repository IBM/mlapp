# Part 4 - AutoML

In this part of the Advanced Course we'll explore the options of the AutoML utility of MLApp.

As of now, MLApp supports automatic search for the best model from a variety of estimator families:

- classification
- clustering
- linear 
- non linear

These functions are highly configurable, such as choosing the parameters for each model, hyper-parameters, scoring function, cross validation and more.

It is recommended to go through the documentation as there is thorough explanation on each configuration option.

> Note: these functions also support non-sklearn models as long as they support the interface of the sklearn models (functions such as `fit`, `predict`, etc.)
>
> Example models from other libraries that can also be used: XGBoost, LightGBM.

In this part of the Advanced Topics, we'll see how we use a neural network model with MLApp's AutoML, which is not included in the default models provided by MLApp.


## 2. Prerequisites - Crash Course Asset

Install the crash course with the MLApp's CLI **boilerplates** command in the root of your project and rename it to `mlp_classifier`:
```bash
mlapp boilerplates install crash_course --new-name mlp_classifier
```

Run a train Pipeline via `run.py` and see it has no errors:
```json
{
    "asset_name": "mlp_classifier",
    "config_path": "assets/mlp_classifier/configs/mlp_classifier_train_config.yaml",
    "config_name": "mlp_classifier_config"
}
```

## 3. MLPClassifier

We'll be using the neural network model provided by sklearn: [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). 

Let's update the Model Manager of the MLP Classifier `assets > mlp_classifier > mlp_classifier_model_manager.py`:

## 4. Scaler

First we need to scale our data for the neural network model to fit properly.

Add import of the scaler and pandas library:
```python
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
```

In the `train_model` Pipeline function add the following between the prepare data for train and the run auto ml:
```python
@pipeline
def train_model(self, data):
    # prepare data for train
    ...

    # fit scaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    self.save_object('scaler', scaler)
    
    # run auto ml
    ...
    ...
    ...
```

We are saving the scaler object so when using the forecast pipeline we can use the same transformation of scaling on new data. In order to do so we'll need to add this code to the `forecast` Pipeline function:

```python
# use scaler
scaler = self.get_object('scaler')
data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

# transform X_train and X_test
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
```

## 5. AutoML - AutoMLPandas

Add import of the neural network:
```python
from sklearn.neural_network import MLPClassifier
```

Now in order to use the **MLPClassifier** in the AutoMLPandas lets update the arguments this way:
```python
# run auto ml
result = AutoMLPandas(
    'multi_class', models=["NN"], model_classes={"NN": MLPClassifier},
    fixed_params={
        "NN": {
            "solver": "lbfgs",
            "max_iter": 5000
        }
    },
    hyper_params={
        "NN": {
            "alpha": [0.1, 1],
            "hidden_layer_sizes": [(16, 32), (32, 64), (64, 128)]
        }
    }).run(X_train_scaled, y_train, X_test_scaled, y_test)
```

!!! note "AutoML Utility" 
    
    for explanation on each key in the `AutoMLPandas` check the [AutoML API Reference](/api/utils.automl).
 
## 6. Investigate results

We have sent a few hyper-parameters, lets use the AutoMLResult to view the full results to see which ones were chosen: 

Update the following now with `full=True`:
```python
result.print_report(full=True)
```

The Pipeline function of `train_model` in `assets > mlp_classifier > mlp_classifier_model_manager.py` should look like this:
```python
@pipeline
def train_model(self, data):
    # prepare data for train
    X, y = data.drop(self.model_settings['target'], axis=1), data[self.model_settings['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.model_settings.get('train_percent'))

    # fit scaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    self.save_object('scaler', scaler)

    # transform X_train and X_test
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    # run auto ml
    result = AutoMLPandas(
        'multi_class', models=["NN"], model_classes={"NN": MLPClassifier},
        fixed_params={
            "NN": {
                "solver": "lbfgs",
                "max_iter": 5000
            }
        },
        hyper_params={
            "NN": {
                "alpha": [0.1, 1],
                "hidden_layer_sizes": [(16, 32), (32, 64), (64, 128)]
            }
        }).run(X_train_scaled, y_train, X_test_scaled, y_test)

    # print report
    result.print_report(full=True)

    # save results
    self.save_automl_result(result)
```

## 7. Run the model
Run a train Pipeline via `run.py` and see which hyper-parameters were chosen and compare the results with the previous results you got:
```json
{
    "asset_name": "crash_course",
    "config_path": "assets/crash_course/configs/crash_course_train_config.yaml",
    "config_name": "crash_course_config"
}
```

!!! note "Configuration-Based Assets" 
    
    Remember, it is highly recommended to expose the fixed params and hyper params of the neural network model to the configuration for easy customization on production/cloud environment.

