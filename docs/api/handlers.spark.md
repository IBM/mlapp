# Spark Handler

### load_model
```python
SparkInterface.load_model(self, file_path, module)
```

Loads a spark model

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>file_path:</b></i> path to spark model file
<br/><b><i>module:</b></i> name of module to load
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> None
</td></tr></table>
<br/>

### exec_query
```python
SparkInterface.exec_query(self, query, params=None)
```

Executes Query in the database.

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>query:</b></i> str - query to be executed.
<br/><b><i>params:</b></i> list - list of parameters to be used if necessary in query
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> result of query
</td></tr></table>
<br/>

### load_csv_file
```python
SparkInterface.load_csv_file(self, file_path, sep=',', header=True, toPandas=False, **kwargs)
```

This function reads a csv file and return a spark DataFrame

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>file_path:</b></i> path to csv file
<br/><b><i>sep:</b></i> separator of csv file
<br/><b><i>header:</b></i> include header of file
<br/><b><i>toPandas:</b></i> to load as pandas DataFrame
<br/><b><i>kwargs:</b></i> other keyword arguments containing additional information
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> spark DataFrame (or pandas)
</td></tr></table>
<br/>

