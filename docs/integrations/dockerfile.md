# Dockerfile

Any asset built within a ML App Project can easily be dockerized.
 
## 1. Dockerize Your ML App Project

#### 1.1. Create a Dockerfile

Create the following `Dockerfile` in the root of your project:
```text
FROM python

# Install anything you need here in your docker image


# Creating Application Source Code Directory
RUN mkdir -p /worker

# Setting Home Directory for containers
WORKDIR /worker

# Installing python dependencies
COPY requirements.txt /worker/
RUN pip install -r requirements.txt

# Copying src code to Container
COPY . /worker

# Switching to non-root user
RUN useradd appuser && chown -R appuser /worker

# Add any permissions to the user here
USER appuser

# Exposing Ports
EXPOSE 5035

# Setting Persistent data
VOLUME ["/app-data"]

# Running Python Application
CMD [ "python", "./app.py" ]
```

> Note: make sure you have a `requirements.txt` file containing `mlapp` or otherwise use `pip install mlapp` instead in the Dockerfile.

The Docker image will be using the `app.py` as it's entry point and will communicate with incoming requests by listening to a message queue.

#### 1.2. Build the Docker Image

Go to the root directory of your ML App project where you've built your asset/s and run the following command:

```bash
docker build -t mlapp-worker:1.0.0 .
```

That's it now you can use this Docker up in any environment that supports it!