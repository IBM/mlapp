FROM python:3.8

RUN mkdir -p /mlapp
COPY . /mlapp

RUN pip install /mlapp