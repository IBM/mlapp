#### Installation
In your terminal, execute the following command to install MLApp:
```
pip install mlapp
```

#### Installing Extras for MLApp
For using extra capabilities of ML App use the `pip install` command and add brackets with the extra you wish to install:

<code>pip install "mlapp[mlcp]"</code> - installs ML App with all libraries that are required for using the Machine Learning Control Panel (in-house control panel).

<code>pip install "mlapp[aml]"</code> - installs ML App with all libraries that are required for using Azure Machine Learning.

> Note: check in `setup.py` the _extras_require_ for more information on 3rd party libraries we use to connect with different services.

#### Initiating a New MLApp Project
You are now ready to start your MLApp project! To do so, navigate to the your new project's folder in your terminal and run the following command:

```
mlapp init
```

This command creates the required files for a MLApp project.

#### Next Steps
A great place to start is the [MLApp Crash Course](/crash-course/introduction).
