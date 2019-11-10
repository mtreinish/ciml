FROM tensorflow/tensorflow:1.6.0-py3
LABEL maintainer Andrea Frittoli (andrea.frittoli@gmail.com)

COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y git python3-tk
RUN sed -i -e 's/^tensorflow/# tensorflow/g' /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
