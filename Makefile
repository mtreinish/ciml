CIML_BASE_IMAGE=ciml-base
CIML_PREDICT_IMAGE=mqtt-predict
MQTT_WAIT_IMAGE=mqtt-wait
MQTT_IMAGE=mqtt
IMAGE_REG=registry.ng.bluemix.net/ciml/
BUILDER=bx cr build -t

all: base wait predict mqtt

base:
	$(BUILDER) $(IMAGE_REG)$(CIML_BASE_IMAGE):1 images/$(CIML_BASE_IMAGE)

wait:
	$(BUILDER) $(IMAGE_REG)$(MQTT_WAIT_IMAGE):1 images/$(MQTT_WAIT_IMAGE)

predict:
	$(BUILDER) $(IMAGE_REG)$(CIML_PREDICT_IMAGE):1 images/$(CIML_PREDICT_IMAGE)

mqtt:
	$(BUILDER) $(IMAGE_REG)$(MQTT_IMAGE):1 images/$(MQTT_IMAGE)
