# Handlers

When adding services to your [environment](/concepts/environment) you can use these services via handlers.

There are four different types of handlers you can use with MLApp depending on the service:

- Database (mysql, postresql, snowflake, etc..)
- File Storage (azure blob, minio, s3, etc..)
- Message Queue (rabbitmq, kafka, azure service bus, etc..)
- Spark (spark, databricks, etc..)

Once you added services to your environment file you can see them via the following command:

```bash
mlapp services show
```

An example output of the MLCP environment file:
```text
Service Name    Service Type
--------------  --------------
RABBITMQ        rabbitmq
MINIO           minio
POSTGRES        postgres
```

To use these services with a handler first import one or more of the following anywhere in your code:
```python
from mlapp.handlers.instance import database_handler, file_storage_handler, message_queue_handler, spark_handler
```

In case we want to use the 'POSTGRES' handler for example just use:
```python
database_handler('POSTGRES').<method>(args)
```

> Note: we have used the service name **POSTGRES** in lower case to fetch the handler for the service. For supported methods for each handler check the [API Reference](/api/handlers.database).

