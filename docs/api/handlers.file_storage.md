# File Storage Handler

### download_file
```python
FileStorageInterface.download_file(self, bucket_name, object_name, file_path, *args, **kwargs)
```

Downloads file from file storage

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>bucket_name:</b></i> name of the bucket/container
<br/><b><i>object_name:</b></i> name of the object/file
<br/><b><i>file_path:</b></i> path to local file
<br/><b><i>args:</b></i> other arguments containing additional information
<br/><b><i>kwargs:</b></i> other keyword arguments containing additional information
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> None
</td></tr></table>
<br/>

### stream_file
```python
FileStorageInterface.stream_file(self, bucket_name, object_name, *args, **kwargs)
```

Streams file from file storage

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>bucket_name:</b></i> name of the bucket/container
<br/><b><i>object_name:</b></i> name of the object/file
<br/><b><i>args:</b></i> other arguments containing additional information
<br/><b><i>kwargs:</b></i> other keyword arguments containing additional information
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> file stream
</td></tr></table>
<br/>

### upload_file
```python
FileStorageInterface.upload_file(self, bucket_name, object_name, file_path, *args, **kwargs)
```

Uploads file to file storage

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>bucket_name:</b></i> name of the bucket/container
<br/><b><i>object_name:</b></i> name of the object/file
<br/><b><i>file_path:</b></i> path to local file
<br/><b><i>args:</b></i> other arguments containing additional information
<br/><b><i>kwargs:</b></i> other keyword arguments containing additional information
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> None
</td></tr></table>
<br/>

### list_files
```python
FileStorageInterface.list_files(self, bucket_name, prefix='', *args, **kwargs)
```

Lists files in file storage

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>bucket_name:</b></i> name of the bucket/container
<br/><b><i>prefix:</b></i> prefix string to search by
<br/><b><i>args:</b></i> other arguments containing additional information
<br/><b><i>kwargs:</b></i> other keyword arguments containing additional information
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> file names list
</td></tr></table>
<br/>

