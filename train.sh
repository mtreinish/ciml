#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

DATASET=${1:-DATASET}
EXPERIMENT=${2:-EXPERIMENT}

# Install any missing depenency
pip install -r requirements.txt;

# Run the training
python3 -c "from ciml.trainer import local_trainer; local_trainer()" \
  --data-path $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT
