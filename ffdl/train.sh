#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

DATASET=${1:-DATASET}
EXPERIMENT=${2:-EXPERIMENT}
GPU=${3:-"false"}
GPU_PARAM=[[ "$GPU" == "true" ]] && echo "--gpu"

# Install CIML requirements (excluded tensorflow)
# The right version of tf is selected in the ffdl manifest
sed -i -e 's/^tensorflow/# tensorflow/g' requirements.txt
pip install -r requirements.txt

# Set the default backend of matplotlib to a non GUI one
mkdir -p ~/.config/matplotlib
echo "backend : agg" > ~/.config/matplotlib/matplotlibrc

# Run the training
python3 -c "from ciml.trainer import local_trainer; local_trainer()" \
  --data-path $DATA_DIR --dataset $DATASET --experiment $EXPERIMENT $GPU_PARAM

# Save the training results
cp -r $MODEL_DIR/data $RESULT_DIR/
