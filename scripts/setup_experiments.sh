#!/usr/bin/env bash

# Setup/refresh all experiments
# It requires CIML to be installed, and aws profile "ibmcloud"
# to be defined.

# Call with --force to force recreating datasets

BATCH=128
TARGET_DATA_PATH=${TARGET_DATA_PATH:-/git/github.com/mtreinish/ciml/data}
S3_AUTH_URL=${S3_AUTH_URL:-https://s3.eu-geo.objectstorage.softlayer.net}
S3_PROFILE=${S3_PROFILE:-ibmcloud}

declare -A NETWORK_NAMES
NETWORK_NAMES["10/10/10"]=dnn-3x10
NETWORK_NAMES["100/100/100"]=dnn-3x100
NETWORK_NAMES["100/100/100/100/100"]=dnn-5x100
NETWORK_NAMES["500/500/500/500/500"]=dnn-5x500
NETWORK_NAMES["100/100/100/100/100/100/100/100/100/100"]=dnn-10x100
NETWORK_NAMES["1000/1000/1000"]=dnn-3x1000

EPOCHS="100 500 1000 5000"

for network in "${!NETWORK_NAMES[@]}"; do
  for epochs in ${EPOCHS}; do
    # Setup the experiment
    EXPERIMENT=${NETWORK_NAMES[$network]}-${epochs}epochs-bs${BATCH}
    echo "=== Setting up experiment $EXPERIMENT"
    ciml-setup-experiment --experiment $EXPERIMENT \
      --estimator tf.estimator.DNNClassifier \
      --hidden-layers $network \
      --steps $(( 2500 / BATCH * epochs )) \
      --batch-size $BATCH \
      --epochs ${epochs} \
      --data-path $TARGET_DATA_PATH \
      --s3-profile $S3_PROFILE \
      --s3-url $S3_AUTH_URL $@
  done
done
