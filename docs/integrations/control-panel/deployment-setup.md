# Control Panel - Deployment Setup

In this guide we will walk through the steps to orchestrate the control panel and your python worker on [Kuberenetes](https://kubernetes.io/).

## Prerequisites
- Kubernetes Cluster.
- Kubernetes CLI.
- Relational Database for storing model metadata.
- File Storage for storing model files.
- Message Queue for communicating between the microservices.

!!! tip "Setting up your Services"

    Unless you're know what you're doing, it is recommended to use a [Helm Chart](https://helm.sh/) if you're deploying your services on Kubernetes, or otherwise use a managed service.

## Setup

#### MLAPP's Control Panel UI Deployment YAML
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlapp-cp-ui
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlapp-cp-ui
  template:
    metadata:
      labels:
        app: mlapp-cp-ui
    spec:
      volumes:
        - name: config-volume
          configMap:
            name: radml-vuejs-configmap        
      containers: 
      - name: mlapp-cp-ui
        imagePullPolicy: Always
        image: ibmcom/mlapp-cp-ui:latest 
        env:
        - name: NODE_ENV
          value: "production"
        ports:
          - containerPort: 8080
        volumeMounts:
          - name: config-volume
            mountPath: /usr/share/nginx/html/env-config.js
            subPath: env-config.js
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlapp-cp-ui-configmap
data:
  env-config.js: |
    var env_config = (() => {
      return {
        "VUE_APP_BASE_URL": "'$INGRESS_URL'/backend",
        "VUE_APP_FILE_STORAGE_URL": null,
        "VUE_APP_LOGIN_REQUIRED": true,
        "VUE_APP_LOGIN_TYPE": "basic",
        "VUE_APP_LOGIN_BACKGROUND": "'$VUE_APP_LOGIN_BACKGROUND_IMG_URL'",
        "VUE_APP_LOGO": "'$VUE_APP_LOGO_IMG_URL'",
        "FILE_STORE_BUCKETS": "imgs:'$VUE_APP_IMGS_BUCKET',configs:'$VUE_APP_CONFIGS_BUCKET',logs:'$VUE_APP_LOGS_BUCKET'"
      };
    })();   
---
apiVersion: v1
kind: Service
metadata:
  name: mlapp-cp-ui
spec:
  ports:
  - port: 8080
    protocol: TCP
    targetPort: 8080
  selector:
    app: mlapp-cp-ui
  type: NodePort
```

#### MLAPP's Control Panel API Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlapp-cp-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlapp-cp-api
  template:
    metadata:
      labels:
        app: mlapp-cp-api
    spec: 
      containers: 
      - name: mlapp-cp-api
        imagePullPolicy: Always
        image: ibmcom/mlapp-cp-api:latest 
        env:
        - name: NODE_ENV
          value: "production"
        - name: CORS
          value: "'$INGRESS_URL'"
        - name: DB_ADAPTER
          value: "postgres"
        - name: DB_TYPE
          value: "knex"
        - name: DB_HOST
          value: "'$DB_HOST'"
        - name: DB_NAME
          value: "'$DB_NAME'"
        - name: DB_USER
          value: "'$DB_USER'"
        - name: DB_PASSWORD
          value: "'$DB_PASSWORD'"
        - name: DB_PORT
          value: "'$DB_PORT'"                   
        - name: DB_SSL
          value: "true"
        - name: MQ_TYPE
          value: "rabbitmq"                   
        - name: MQ_ENDPOINT
          value: "'$MQ_ENDPOINT'"                   
        - name: MQ_CERT_TEXT
          value: "'$MQ_CERT_TEXT'"                   
        - name: SEND_ANALYSIS_TOPIC
          value: "analysis_general_listen"
        - name: RESPONSE_ANALYSIS_TOPIC
          value: "analysis_respond"
        - name: FS_TYPE
          value: "ibm-cos-sdk"  
        - name: BOTO_IBM_BOTO3_ENDPOINT
          value: "'$BOTO_IBM_BOTO3_ENDPOINT'"
        - name: BOTO_IBM_BOTO3_API_KEY_ID
          value: "'$BOTO_IBM_BOTO3_API_KEY_ID'"                   
        - name: BOTO_IBM_BOTO3_SERVICE_INSTANCE_ID
          value: "'$BOTO_IBM_BOTO3_SERVICE_INSTANCE_ID'"     
        - name: SESSION_TYPE
          value: "redis"
        - name: REDIS_HOST
          value: "'$REDIS_HOST'"                                  
        - name: REDIS_PORT
          value: "'$REDIS_PORT'"
        - name: REDIS_PASSWORD
          value: "'$REDIS_PASSWORD'"                                
        - name: REDIS_CERT_TEXT
          value: "'$REDIS_CERT_TEXT'"                                  
        - name: APP_LOGIN_REQUIRED
          value: "true"
        - name: APP_LOGIN_TYPE
          value: "basic"               
        - name: CLIENT_API_TOKEN
          value: "'$CLIENT_API_TOKEN'"
        - name: APP_PKEY
          value: '$APP_PKEY'
        - name: APP_IS_HTTPS
          value: "true"
        ports:
        - containerPort: 3000
---
apiVersion: v1
kind: Service
metadata:
  name: mlapp-cp-api
spec:
  ports:
  - port: 80
    protocol: TCP
    targetPort: 3000
  selector:
    app: mlapp-cp-api
  type: ClusterIP
```

!!! note "Filling out Variables"

    In the examples above, the values are filled with the default Contorl Panel setup. Once you have all your services up and running you can fill up the credentials. 


## Exposing the Control Panel for Outside Access

In order to access the control panel, you will have to expose the services of the Control Panel UI & API.

Example using [IBMCloud](https://cloud.ibm.com/)'s managed k8 service:

```yaml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: ingress-vuejs
spec:
  tls:
  - hosts:
    - <HOST_NAME>
    secretName: <SECRET_NAME>
  rules:
  - host: <HOST_NAME>
    http:
      paths:
      - path: /
        backend:
          serviceName: mlapp-cp-ui
          servicePort: 8080

---
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: ingress-nodejs
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2
spec:
  tls:
  - hosts:
    - <HOST_NAME>
    secretName: <SECRET_NAME>
  rules:
  - host: <HOST_NAME>
    http:
      paths:
      - path: /backend(/|$)(.*)
        backend:
          serviceName: mlapp-cp-api
          servicePort: 80
```

## Setting Up the Python Worker

Follow the instructions [here](/integrations/kubernetes).