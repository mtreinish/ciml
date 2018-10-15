#!/bin/bash

# Usage:
#   export S3_AUTH_URL S3_ACCESS_KEY_ID S3_SECRET_ACCESS_KEY
#   ffdl_train.sh dataset experiment

# Generate the model zip
git archive -o ffdl/ciml.zip HEAD

# Number of GPUs to use
GPUS=${3:-0}

# Generate the manifest file
python ffdl/make-manifest.py \
  --experiment $2 --dataset $1 --gpus $GPUS \
  --s3-access-key-id $S3_ACCESS_KEY_ID \
  --s3-secret-access-key $S3_SECRET_ACCESS_KEY \
  --s3-auth-url $S3_AUTH_URL > ffdl/manifest.yaml

# Expects ffdl to be installed and in the PATH
ffdl train ffdl/manifest.yaml ffdl/ciml.zip
