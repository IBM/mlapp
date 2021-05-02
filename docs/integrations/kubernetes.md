# Kubernetes

## Prerequisites

In order of having your asset deployed in a Kubernetes cluster you must have the following resources:

- Kubernetes Cluster.
- Kubernetes CLI.
- Container Registry containing Docker image: _mlapp-worker_. 
- Relational Database for storing model metadata.
- File Storage for storing model files.
- Message Queue for communicating between the microservices.
- Add to your project's `requirements.txt` any external services you're using.

!!! note "`requirments.txt` libraries used with the default MLApp's Control Panel setup"

    The default external libraries we use are: `pika`, `pg8000<=1.16.5` and `minio`.
 
!!! tip "Setting up your Services"

    Unless you're know what you're doing, it is recommended to use a [Helm Chart](https://helm.sh/) if you're deploying your services on Kubernetes, or otherwise use a managed service.

## Setting up the MLApp Worker

Prepare a `.yaml` file for the MLApp worker:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlapp-worker-config
  labels:
    name: mlapp-worker-config
data:
  config.py: |
    settings = {
      "env_file_path": "env/.env",
       "deployEnvironment": "default",
       "deployVersion": "'"$SOURCE_BUILD_NUMBER"'",
      "queues": {
        "listen_queue_names": ["analysis_general_listen"],
        "send_queue_name": "analysis_respond"
      },
      "file_store_buckets": {
          "objects": "'$OBJECTS_BUCKET'",
          "csvs": "'$CSVS_BUCKET'",
          "configs": "'$CONFIGS_BUCKET'",
          "metadata": "'$METADATA_BUCKET'",
          "imgs": "'$IMGS_BUCKET'",
          "logs": "'$LOGS_BUCKET'"
      }
    }
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlapp-worker-env
  labels:
    name: mlapp-worker-env
data:
  .env: |
    # RabbitMQ
    RABBITMQ_MLAPP_SERVICE_TYPE=rabbitmq
    RABBITMQ_MAIN_MQ=true
    RABBITMQ_RABBITMQ_HOSTNAME='$RABBITMQ_RABBITMQ_HOSTNAME'
    RABBITMQ_RABBITMQ_PORT='$RABBITMQ_RABBITMQ_PORT'
    RABBITMQ_RABBITMQ_USERNAME='$RABBITMQ_RABBITMQ_USERNAME'
    RABBITMQ_RABBITMQ_PASSWORD='$RABBITMQ_RABBITMQ_PASSWORD'
    RABBITMQ_RABBITMQ_CONNECTION_TIMEOUT='$RABBITMQ_RABBITMQ_CONNECTION_TIMEOUT'
    RABBITMQ_RABBITMQ_CERT_PATH=./certs/rabbitmq.pem
    # Postgres
    POSTGRES_MLAPP_SERVICE_TYPE=postgres
    POSTGRES_MAIN_DB=true
    POSTGRES_POSTGRES_DATABASE_NAME='$POSTGRES_POSTGRES_DATABASE_NAME'
    POSTGRES_POSTGRES_HOSTNAME='$POSTGRES_POSTGRES_HOSTNAME'
    POSTGRES_POSTGRES_PORT='$POSTGRES_POSTGRES_PORT'
    POSTGRES_POSTGRES_USER_ID='$POSTGRES_POSTGRES_USER_ID'
    POSTGRES_POSTGRES_PASSWORD='$POSTGRES_POSTGRES_PASSWORD'
    POSTGRES_POSTGRES_SSL=true
    # minio service
	MINIO_MLAPP_SERVICE_TYPE=minio
	MINIO_MAIN_FS=true
	MINIO_MINIO_ENDPOINT='$MINIO_MINIO_ENDPOINT'
	MINIO_MINIO_ACCESS_KEY='$MINIO_MINIO_ACCESS_KEY'
	MINIO_MINIO_SECRET_KEY='$MINIO_MINIO_SECRET_KEY'
	MINIO_MINIO_PORT='$MINIO_MINIO_PORT'
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlapp-worker
spec:
  selector:
    matchLabels:
      app: mlapp-worker
  replicas: 1
  template:
    metadata:
      labels:
        app: mlapp-worker
    spec:
      volumes:
        - name: mlapp-worker-env-volume
          configMap:
            name: mlapp-worker-env
        - name: mlapp-worker-config-volume
          configMap:
            name: mlapp-worker-config
        - name: rabbitmq-cert
          secret:
            secretName: mlapp-rabbitmq-cert
      containers:
      - name: mlapp-worker
        tty: true
        imagePullPolicy: Always
        image: $IMAGE_REPOSITRY_URL
        command: ["python3", "./app.py"]
        args: ["while true; do sleep 30; done;"]
        ports:
        - containerPort: 5035
        volumeMounts:
        - name: mlapp-worker-env-volume
          mountPath: /worker/env/.env
          subPath: .env
        - name: mlapp-worker-config-volume
          mountPath: /worker/config.py
          subPath: config.py
        resources:
            requests:
              memory: "2Gi"
              cpu: "1"
            limits:
              memory: "4Gi"
              cpu: "2"
``` 

!!! note "Deploying `.env` and `config.py` Files"

    `.env` and `config.py` files are needed to be filled. 

    In the example above, the values are filled with the default Contorl Panel setup. Once you have all your services up and running you can fill up the credentials and save them as [ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/) or [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/) in your Kuberenetes cluster and mount them into your MLApp worker deployment. 

    

## Create The YAML

Run the following command to create all deployments, services and configmaps:

```bash
kubectl create -f <path_to_yaml_file>.yaml
```

Once finished you should see all your pods up and running via command:
```bash
kubectl get pods
```