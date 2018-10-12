#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

DATASET=${1:-DATASET}
EXPERIMENT=${2:-EXPERIMENT}

# Install any missing depenency
grep -v tensorflow requirements.txt > notf_requirements.txt
pip install -r notf_requirements.txt
apt-get update
apt-get install -y python3.5-tk

# Run the training
python3 -c "from ciml.trainer import local_trainer; local_trainer()" \
  --data-path $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --gpu

# Save the training results
cp -r $MODEL_DIR/data $RESULT_DIR/
