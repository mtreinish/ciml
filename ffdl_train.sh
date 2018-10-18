#!/bin/bash

# Usage:
#   export S3_AUTH_URL AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
#   ffdl_train.sh dataset experiment

CIML_DIR=$(cd $(dirname $0); pwd)
RUN_DIR=$(pwd)

# Generate the model zip
pushd "$CIML_DIR"
mkdir -p "${RUN_DIR}/ffdl"
git archive -o "$RUN_DIR/ffdl/ciml.zip" HEAD
popd

# Number of GPUs to use
GPUS=${3:-0}

# Generate the manifest file
python "$CIML_DIR/ffdl/make-manifest.py" \
  --experiment $2 --dataset $1 --gpus $GPUS \
  --s3-access-key-id $AWS_ACCESS_KEY_ID \
  --s3-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --s3-auth-url $S3_AUTH_URL > ffdl/manifest.yaml

# Expects ffdl to be installed and in the PATH
ffdl train ffdl/manifest.yaml ffdl/ciml.zip
