CIML_BASE_IMAGE=ciml-base
CIML_PREDICT_IMAGE=mqtt-predict
MQTT_WAIT_IMAGE=mqtt-wait
CIML_API_IMAGE=ciml-api
MQTT_IMAGE=mqtt
IMAGE_REG=registry.ng.bluemix.net/ciml/
BUILDER=bx cr build -t

all: base wait predict mqtt api

base:
	$(BUILDER) $(IMAGE_REG)$(CIML_BASE_IMAGE):1 images/$(CIML_BASE_IMAGE)

wait:
	$(BUILDER) $(IMAGE_REG)$(MQTT_WAIT_IMAGE):1 images/$(MQTT_WAIT_IMAGE)

predict:
	$(BUILDER) $(IMAGE_REG)$(CIML_PREDICT_IMAGE):1 images/$(CIML_PREDICT_IMAGE)

mqtt:
	$(BUILDER) $(IMAGE_REG)$(MQTT_IMAGE):1 images/$(MQTT_IMAGE)

api:
	$(BUILDER) $(IMAGE_REG)$(CIML_API_IMAGE):1 images/$(CIML_API_IMAGE)
