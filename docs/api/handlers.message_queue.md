# Message Queue Handler

### send_message
```python
MessageQueueInterface.send_message(self, queue_name, body)
```

Sends message to the queue

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>queue_name:</b></i> name of the topic/queue to send the message to
<br/><b><i>body:</b></i> message as string or bytes
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> None
</td></tr></table>
<br/>

### listen_to_queues
```python
MessageQueueInterface.listen_to_queues(self, queue_names, callback)
```

Listen to queues/topics

<br/><table style="width:100%"><tr><td valign="top"><b><i>Parameters:</b></i></td><td valign="top"><b><i>queue_names:</b></i> list of queue/topic names to listen to
<br/><b><i>callback:</b></i> function to call upon receiving a message
</td></tr><tr><td valign="top"><b><i>Returns:</b></i></td><td valign="top"> None
</td></tr></table>
<br/>

