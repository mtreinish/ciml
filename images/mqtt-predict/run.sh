#!/bin/sh

export MPLBACKEND=agg
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

ciml-mqtt-predict --mqtt-hostname $MQTT_HOST --topic='#' --sample-interval "1min"
