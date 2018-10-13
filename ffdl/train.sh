#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

DATASET=${1:-DATASET}
EXPERIMENT=${2:-EXPERIMENT}

# Install CIML
pip install .

# Run the training
ciml-train-model --data-path $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT --gpu

# Save the training results
cp -r $MODEL_DIR/data $RESULT_DIR/
