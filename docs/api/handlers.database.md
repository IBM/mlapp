# Database Handler

### execute_query
```python
DatabaseInterface.execute_query(self, query, params=None)
```

Executes Query in the database.

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>query:</b></i> str - query to be executed.
<br/><b><i>params:</b></i> list - list of parameters to be used if necessary in query
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> result of query
</td></tr></table>
<br/>

### insert_query
```python
DatabaseInterface.insert_query(self, query, values)
```

Executes an "INSERT" query in the database.

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>query:</b></i> str - query to be executed.
<br/><b><i>values:</b></i> list - list of values to be used in the query
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> None
</td></tr></table>
<br/>

### insert_df
```python
DatabaseInterface.insert_df(self, sql_table, df, batch_length=1000)
```

Inserts a DataFrame into a table in the database.

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>sql_table:</b></i> str - name of the table.
<br/><b><i>df:</b></i> DataFrame (Pandas, PySpark or other) - Matrix type DataFrame containing all values to insert.
<br/><b><i>batch_length:</b></i> int - length of the how many rows to insert from matrix at a time
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> None
</td></tr></table>
<br/>

### get_df
```python
DatabaseInterface.get_df(self, query, params=None)
```

Executes a query in the database and returns it as a DataFrame.

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>query:</b></i> str - query to be executed.
<br/><b><i>params:</b></i> list - list of parameters to be used if necessary in query
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> result of query as a DataFrame
</td></tr></table>
<br/>
