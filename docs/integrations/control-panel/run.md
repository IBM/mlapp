# Run

After you have some assets ready in _MLApp_, and [registered](/integrations/control-panel/assets) any of them in the **Assets** page you can send **Jobs** to run their pipelines.

# 1. Run an asset via Job
Simply hit the **Run 1 Job** button to do so:
![run-screen](/integrations/control-panel/imgs/run-screen.png)

You can switch between the assets you registered if you're interested in loading a different template configuration:
![run-screen-switch](/integrations/control-panel/imgs/run-screen-switch.png)

# 2. Edit Template
You can click any tab in the setup wizard (**Data Settings**, **Model Settings** or **Job Settings**) to easily edit parts of the configuration:
![settings-split](/integrations/control-panel/imgs/run-config-split.png)

# 3. Configuration AutoGen

If you're interested in taking the full advantage of the configuration based asset you've built, you can generate multiple configurations using the AutoGen:
![run-screen-autogen](/integrations/control-panel/imgs/run-screen-autogen.png)

Here you can add a JSON that will instruct the combination of values you want to add to your configuration.

Lets say we want to check our model with different train/test split point, we can add this JSON:
```json
{
  "models_settings.train_percent": [0.7, 0.8, 0.9]
}
```

Notice that the key is the path to the nested key of the configuration, and the value is a list of different values you want to try.

In the example above we are trying a split for train and test in different points (70%/80%/90%).

Adding more keys and values to the AutoGen JSON will create all the combinations available.

When done click the **Add** button:
![configuration-autogen](/integrations/control-panel/imgs/run-configure-autogen.png)

You will notice the number of jobs getting sent increased by the number of combinations. In the example above it creates 3 different jobs:
![autogen-added-jobs](/integrations/control-panel/imgs/run-autogen-jobs.png)