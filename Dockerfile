FROM python:3.8

COPY model_requirements.txt ./
RUN pip install -r model_requirements.txt
WORKDIR /opt
ADD * /opt/runner/
