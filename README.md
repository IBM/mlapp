# IBM Services Framework for ML Applications (MLApp)
IBM Services Framework for ML Applications (MLApp) is a Python 3 framework for building machine learning applications that are robust and production-ready. It was developed by IBMers based on learnings from dozens of machine learning projects for IBM clients.

IBM Services Framework for ML Applications is the official RAD-ML component accelerator for analytics applications. RAD-ML is a proven methodology for developing sellable, reusable, and scalable machine learning assets.

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
Check out the [Documentation](http://mlapp-docs.apic.mybluemix.net).

A great place to start is the [ML App Crash Course](http://mlapp-docs.apic.mybluemix.net/crash-course/introduction).

#### Contributing to MLApp
We welcome contributions from the community to this framework. Please refer to [CONTRIBUTING](./CONTRIBUTING.md) for more information.

#### **Main Authors**
*  [Tomer Galula](mailto:tomer.galula@ibm.com)  
*  [Tal Waitzenberg](mailto:tal.waitzenberg@ibm.com)  
*  [Erez Nardia](mailto:erez.nardia@ibm.com)
*  [Michael Chein](mailto:michael.chein@ibm.com)
*  [Annaelle Cohen](mailto:annaelle.cohen@ibm.com)  
*  [Nathan Katz](mailto:katzn@us.ibm.com)
*  Keren Haddad-Leibovich  
