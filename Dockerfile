FROM python:3.8

RUN pip install redis
WORKDIR /opt
ADD ./* /opt/runner/
