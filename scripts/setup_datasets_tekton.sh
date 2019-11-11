#!/usr/bin/env bash

# Setup/refresh all datasets required for various experiments
# It requires a k8s configuration in place.
# The target cluster should have Tekton installed, the ciml-create-dataset Task
# deployed, and the s3 secret in place to access the used buckets.
# It assumes enough examples are cached already.
# It assumes a git Tekton resource exists in the cluster.

# Call with -p force=True to force recreating datasets

DATA_PATH=${DATA_PATH:-s3://cimlrawdata}
TARGET_DATA_PATH=${TARGET_DATA_PATH:-s3://cimlodsceu2019}
SLICE=${SLICE:-":5000"}
GIT_RESOURCE=${GIT_RESOURCE:-ciml-master}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-20}

function _running_tasks() {
  tkn tr list | egrep -c -v '(STATUS|Succeeded|Failed|ciml-run-training)'
}

function _wait_for_slot(){
  running=$(_running_tasks)
  while [[ $running -ge $MAX_CONCURRENCY ]]; do
    echo "Waiting: $running tasks"
    sleep 5
    running=$(_running_tasks)
  done
}

function create_datasets() {
  for feature_regex in ${FEATURES}; do
    for sampling in ${SAMPLINGS}; do
      for class_label in ${CLASS_LABELS}; do
        for build_name in ${BUILD_NAMES}; do
          DATASET=$(echo $feature_regex | tr "|" "_" | sed -e "s/(//g" -e "s/)//g")-${sampling}-${class_label}
          if [[ "$build_name" == "tempest-full-py3" ]]; then
            DATASET="${DATASET}-py3"
          fi
          echo "=== Setting up dataset $DATASET"
          # Wait for a slot, do not overload the cluster
          _wait_for_slot
          # Build the dataset
          tkn task start ciml-create-dataset \
            -i ciml=$GIT_RESOURCE \
            -p dataset=$DATASET \
            -p build-name=$build_name \
            -p slicer=$SLICE \
            -p sample-interval="$sampling" \
            -p features-regex="$feature_regex" \
            -p class-label=$class_label \
            -p tdt-split="6 2 2" \
            -p data-path=$DATA_PATH \
            -p target-data-path=$TARGET_DATA_PATH \
            --showlog=false \
            --nocolour \
            -p force=true $@
        done
      done
    done
  done
}

# Dataset by feature/label
FEATURES="(usr|used|1m) (usr|1m) (usr|used) (usr) (used) (1m)"
CLASS_LABELS="node_provider_all node_provider status"
SAMPLINGS="1min"
BUILD_NAMES="tempest-full tempest-full-py3"
create_datasets

# Dataset by sampling/label
FEATURES="(usr|1m)"
CLASS_LABELS="node_provider_all node_provider status"
SAMPLINGS="1s 10s 30s 1min 5min 10min"
BUILD_NAMES="tempest-full tempest-full-py3"
create_datasets
