FROM openwhisk/python3action:latest
MAINTAINER Andrea Frittoli (andrea.frittoli@gmail.com)

COPY requirements-no-tf.txt /tmp/requirements.txt
RUN apk add alpine-sdk libpng-dev freetype-dev
RUN pip install -r /tmp/requirements.txt
