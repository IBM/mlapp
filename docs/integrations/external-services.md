# External Services

When developing an asset with the MLApp framework, more than often you will need to connect with external services.

MLApp framework can be configured and connect with your external services.

### Environment

Most of the time these external services require secret credentials to access them. You can use `.env` files that hold these secrets and are excluded from any version control system.

You can manage multiple `.env` files, e.g. for local development and cloud.

### Initiate an Environment File

In your project you can initiate your first environment file by using the following command:
```text
Usage: mlapp environment init [OPTIONS] NAME

  Usage:

  `mlapp environment init` - creates an empty env file with default name
  '.env'.

  `mlapp environment init NAME` - creates an empty env file named NAME.

Options:
  -h, --help  Show this message and exit.
```

### Changing Environments

You can change your current environment by using the following command:
```text
Usage: mlapp environment set [OPTIONS] NAME

  Usage:

  `mlapp environment set NAME` - sets MLApp to point on NAME
  environment file, (this command modifies config.py file).

Options:
  -h, --help  Show this message and exit.
```
!!! note "env_file_path"

    This will change the `env_file_path` configuration in the `config.py` file in your MLApp project.

### Managing Services in Your Environment

You can manage your services with the following command:
```text
Usage: mlapp services [OPTIONS] COMMAND [ARGS]...

  ML App Services Command

Options:
  -h, --help  Show this message and exit.

Commands:
  add         Usage: `mlapp services add SERVICE_NAME` - register a new service to MLApp 
  delete      Usage: `mlapp services delete SERVICE_NAME` - deletes a registered service
  show        Usage: `mlapp services show` - shows all your registered services.
  show-types  Usage: `mlapp services show-types` -  shows all MLApp's supported services.
```

### Adding a Service Example

The following will show an example of adding a **PostgreSQL** service:

Run the following command:
```bash
mlapp services add postgres
```

This will prompt for the service name:
```text
Please name your service (to access the service in the code): postgres
```

Afterwards you will be prompt for whether it's your main database:
```text
Is it your main database (Y/N, default yes): N
```

!!! tip "Main Handlers"

    Setting a service as a **main** handler makes it act differently, for example: a database service set as **main database** will save all metadata in the database automatically. Same thing applies for a file storage service, setting it as main will save all files in the file storage instead of your local file system.

    For more information check the [full application example](/integrations/external-services/full-app-example).


Afterwards you will be prompt for the service connection credentials:
```text
Hostname (Press enter to set hostname as localhost): 
Port (Press enter to set port to 5432): 
Database name (Enter your database name): mlapp
Username (Enter your database user id): root
Password (Enter your database password): password
```

!!! note "Using Default Values"
    
    Pressing enter will use the default value.

When done, this will add the service connection credentials in the environment file that is set in your project.

In your environment file you will see the following text added:
```text
# postgres postgres service
POSTGRES_MLAPP_SERVICE_TYPE=postgres
POSTGRES_MAIN_DB=false
POSTGRES_POSTGRES_HOSTNAME=127.0.0.1
POSTGRES_POSTGRES_PORT=5432
POSTGRES_POSTGRES_DATABASE_NAME=mlapp
POSTGRES_POSTGRES_USER_ID=root
POSTGRES_POSTGRES_PASSWORD=password
```

!!! note "Caution" 
    
    The `.env` file shouldn't be manually changed, as MLApp expects it in a certain format.

There are four different types of handlers you can use with MLApp depending on the service:

- Database (mysql, postresql, snowflake, etc..)
- File Storage (azure blob, minio, s3, etc..)
- Message Queue (rabbitmq, kafka, azure service bus, etc..)
- Spark (spark, databricks, etc..)

Once you added services to your environment file you can see them via the following command:

```bash
mlapp services show
```

An example output of the Control Panel environment file:
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

!!! tip "API Reference" 

    For supported methods for each handler check the [API Reference](/api/handlers.database).

