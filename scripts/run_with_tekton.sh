#!/usr/bin/env bash
set -e

GIT_RESOURCE=${GIT_RESOURCE:-ciml-master}
EXPERIMENTS_BUCKET_RESOURCE=${EXPERIMENTS_BUCKET_RESOURCE:-cimlexperiments}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-20}
DATASETS=${DATASETS:-datasets}
EXPERIMENTS=${EXPERIMENTS:-experiments}
DATA_BUCKET=${DATA_BUCKET:-cimlodsceu2019}
OUTPUT_BUCKET=${OUTPUT_BUCKET:-cimlodsceu2019output}
TRAINING_LOG_PATH=${TRAINING_LOG_PATH:-.}

function _running_tasks() {
  kubectl get tr -l "ciml/run.uuid=$RUN_UUID"  2> /dev/null | egrep -c -v '(NAME|Succeeded|Failed)' || true
}

function _wait_for_slot(){
  running=$(_running_tasks)
  while [[ $running -ge $MAX_CONCURRENCY ]]; do
    echo "Waiting: $running tasks"
    sleep 5
    running=$(_running_tasks)
  done
}

function _create_bucket_resource() {
  BUCKET=$1
  BPATH=$2
  BNAME=bucket-$(echo $BPATH | tr '_' '-')
  cat <<EOF | kubectl apply -f - -o jsonpath='{.metadata.name}'
  apiVersion: tekton.dev/v1alpha1
  kind: PipelineResource
  metadata:
    name: $BNAME
    labels:
      ciml/run.uuid: $RUN_UUID
  spec:
    params:
    - name: type
      value: gcs
    - name: location
      value: s3://$BUCKET/$BPATH
    - name: dir
      value: "y"
    secrets:
    - fieldName: BOTO_CONFIG
      secretKey: boto_config
      secretName: cos
    type: storage
EOF
}

function run_trainings() {
  mkdir -p ${TRAINING_LOG_PATH}/${RUN_UUID}
  for dataset in $(cat ${DATASETS}); do
    for experiment in $(cat ${EXPERIMENTS}); do
      if [ "$dataset" == "__experiments__" ]; then
        dataset=${experiment%%;*}
        experiment=${experiment#*;}
      fi
      echo "=== Running training $dataset / $experiment"
      # Wait for a slot, do not overload the cluster
      _wait_for_slot
      # Generate UUID and create the log
      UUID=$(uuidgen | tr '[:upper:]' '[:lower:]')
      echo "$(date +'%F %R');$dataset;$experiment;$UUID" >> ${TRAINING_LOG_PATH}/${RUN_UUID}/training_log.csv
      # Create Tekton resources
      DATASET_BUCKET_RESOURCE=$(_create_bucket_resource $DATA_BUCKET $dataset)
      OUTPUT_BUCKET_RESOURCE=$(_create_bucket_resource $OUTPUT_BUCKET $UUID)
      # Debug the training command
      echo tkn task start \
        -i ciml=$GIT_RESOURCE \
        -i experiments=$EXPERIMENTS_BUCKET_RESOURCE \
        -i dataset=$DATASET_BUCKET_RESOURCE \
        -o cimloutput=$OUTPUT_BUCKET_RESOURCE \
        -p dataset=$dataset \
        -p experiment=$experiment \
        -s ciml-bot \
        --showlog=false \
        --labels ciml/run.uuid=$RUN_UUID \
        --nocolour \
        ciml-run-training
      # Start the training
      tkn task start \
        -i ciml=$GIT_RESOURCE \
        -i experiments=$EXPERIMENTS_BUCKET_RESOURCE \
        -i dataset=$DATASET_BUCKET_RESOURCE \
        -o cimloutput=$OUTPUT_BUCKET_RESOURCE \
        -p dataset=$dataset \
        -p experiment=$experiment \
        -s ciml-bot \
        --showlog=false \
        --labels ciml/run.uuid=$RUN_UUID \
        --nocolour \
        ciml-run-training
    done
  done
}

RUN_UUID=$(uuidgen | tr '[:upper:]' '[:lower:]')
run_trainings
