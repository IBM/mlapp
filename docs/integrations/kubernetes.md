# Kubernetes

## 1. Prerequisites

In order of having your asset deployed in a Kubernetes cluster you must have the following resources:

- Kubernetes Cluster.
- Kubernetes CLI.
- Container Registry containing Docker image: _mlapp-worker_. 
- Relational Database for storing model metadata.
- File Storage for storing model files.
- Message Queue for communicating between the microservices.

Once you have all that prepared you can start the deployment to Kubernetes.
 
## 2. Setting up the MLApp Worker

Prepare a `.yaml` file for the MLApp worker:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: radml-mlapp-worker-env	# configmap name
  labels:
    name: radml-mlapp-worker-env	# configmap name
data:
  .env: |
    # Copy of configured .env file
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: radml-mlapp-worker-config	# configmap name
  labels:
    name: radml-mlapp-worker-config	# configmap name
data:
  config.py: |
    # Copy of configured config.py file
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlapp-worker	# container name
spec:
  replicas: 3	# number of instances for the container
  template:
    metadata:
      labels:
        app: mlapp-worker	# container name
    spec:
      volumes:
        - name: radml-mlapp-worker-env-volume	# creating configmap volume
          configMap:
            name: radml-mlapp-worker-env	# selecting created configmap
        - name: radml-mlapp-worker-config-volume	# creating configmap volume
          configMap:
            name: radml-mlapp-worker-config	# selecting created configmap
      containers:
      - name: mlapp-worker	# container name
        tty: true
        imagePullPolicy: Always
        image: container-registry/mlapp-worker:latest	# path to image in the Container Registry
        ports:
        - containerPort: 5035
        volumeMounts:
        - name: radml-mlapp-worker-env-volume	# selecting created configmap volume
          mountPath: /worker/env/.env	# path to mount configmap
          subPath: .env	# name of configmap file
        - name: radml-mlapp-worker-config-volume  # selecting created configmap volume
          mountPath: /worker/config.py	# path to mount configmap
          subPath: config.py	# name of configmap file
        resources:
          requests:
            memory: "2Gi"   # Memory minimum required allocation
            cpu: "1"   # CPU minimum required allocation
          limits:
            memory: "4Gi"   # Memory maximum allowed allocation
            cpu: "2"	# CPU maximum allowed allocation
``` 

> Note: `.env` and `config.py` files are needed to be filled. 

## 3. Create The YAML

Run the following command to create all deployments, services and configmaps:

```bash
kubectl create -f <mlapp_worker>.yaml
```

Once finished you should see all your pods up and running via command:
```bash
kubectl get pods
```