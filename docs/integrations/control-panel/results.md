# Results

### 1. View Model Results

After you've trained a model or more, you can make an analysis on the results.

Open the **Results** page in the navigation:
![results-screen](/integrations/control-panel/imgs/results-screen.png)

You can see all the models trained.

### 2. Filter By Asset & Asset Label

In the header of the results page there are filtering options where you can filter by a specific asset and see only it's results:
![results-filter](/integrations/control-panel/imgs/results-filter.png)

It is also possible to filter by an asset label. In case you have an asset that can be run on different scenario, e.g. different cities, you can label each city differently and then filter the results by the city as well.

In order to label a model just add the `asset_label` key in the `job_settings` of the configuration:
```json
{
  "asset_name": "crash_course",
  "pipeline": "train",
  "asset_label": "New York"
}
```  

### 3. Add Score Column

After you setup your desired filters, you can compare between different models you produced. Simply add columns of scores you're intrested:
![add-score-column](/integrations/control-panel/imgs/results-kpi.png)

> Note: in order for your score to show up in this drop-down just add it to the metadata of the model under the `scores` key or use the `self.save_automl_result` method in the Model Manager.
  
### 4. Sort By Score Column

You can then sort by the score you're interested:
![sort-by-score](/integrations/control-panel/imgs/results-sort.png)

In this example we added a score column of `f1 score (test set)` and clicked the head of the column to sort it.

### 5. Model Popup

When you click a row in the results table - a popup will show with more information on the model.

### 5.1. Model's Output
In the **Model Output** tab you can view all the metadata you saved for the model. In the following images you can see an example:
![model-output](/integrations/control-panel/imgs/model-output.png)

### 5.2. Model's Coefficients
In the **Coefficients** tab you can each coefficient of the model and it's value:
![model-coefficients](/integrations/control-panel/imgs/model-coefficients.png)

> Note: in order for your coefficients to show up here just add it to the metadata of the model under the `coefficients` key or use the `self.save_automl_result` method in the Model Manager.

### 5.3. Model's Config
In the **Config** tab you view the configuration used for the model:
![model-config](/integrations/control-panel/imgs/model-config.png)

### 5.4. Model's Images
In the **Images** tab you view the figures you saved:
![model-images](/integrations/control-panel/imgs/model-images.png)

> Note: in order for your images to show up here just save them (matplotlib figures) using the `self.save_image` or `self.save_images` methods in the Data Manager or Model Manager.

### 5.5. Model's Logs
In the **Logs** tab you can view the console output as you would see it when running the asset in your IDE:
![model-log](/integrations/control-panel/imgs/model-log.png)

### 5.6. Model's Actions
In the **Actions** tab you have two available actions, **Select as Best Model** and **Run Pipeline**:
![model-actions](/integrations/control-panel/imgs/model-actions.png)

#### 5.6.1. Model Promotion
When you click the **Select as Best Model** button, you are promoting the current model to be the chosen model for the current asset and asset label. This can later be used for exposing a forecast API using the chosen/promoted model.


#### 5.6.2. Run Pipeline
In this action you can select a pipeline such as **Forecast** or **Retrain** using the current model.

You also have a JSON editor to edit the **Forecast**/**Retrain** configuration being sent.

When ready click the **Run** button to send a configuration.



