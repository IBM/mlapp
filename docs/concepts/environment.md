# Environment

When developing an asset with the MLApp framework, more than often you will need to connect to external resources and services.

In order to do that, we have simplified this process and it can all be controlled via the MLApp CLI.

Most of the time these external services require secret credentials to access them. You can use `.env` files that hold these secrets and are these files are excluded from any version control system.

You can manage multiple `.env` files, e.g. for local development and cloud.

## 1. Initiate an Environment File

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

## 2. Changing Environments

You can change your current environment by using the following command:
```text
Usage: mlapp environment set [OPTIONS] NAME

  Usage:

  `mlapp environment set NAME` - sets ML App to point on NAME
  environment file, (this command modifies config.py file).

Options:
  -h, --help  Show this message and exit.
```

> Note: this will change the `env_file_path` configuration in the `config.py` file in your MLApp project.

## 3. Managing Services in Your Environment

You can manage your services with the following command:
```text
Usage: mlapp services [OPTIONS] COMMAND [ARGS]...

  ML App Services Command

Options:
  -h, --help  Show this message and exit.

Commands:
  add         Usage: `mlapp services add SERVICE_NAME` - register a new service to ML App 
  delete      Usage: `mlapp services delete SERVICE_NAME` - deletes a registered service
  show        Usage: `mlapp services show` - shows all your registered services.
  show-types  Usage: `mlapp services show-types` -  shows all ML App's supported services.
```

## 4. Adding a Service Example

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

> Note: settings this as _Yes_ will save all model metadata in this database.
>
>> Note: For a file storage service, setting main as _Yes_ will save all files in the file storage instead of your local file system.


Afterwards you will be prompt for the service connection credentials:
```text
Hostname (Press enter to set hostname as localhost): 
Port (Press enter to set port to 5432): 
Database name (Enter your database name): mlapp
Username (Enter your database user id): root
Password (Enter your database password): password
```

> Note: pressing enter will use the default value.

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

> Note: the `.env` file shouldn't be manually changed as it can corrupt the order of the file MLApp expects.
