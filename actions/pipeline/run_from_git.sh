#!/usr/bin/env bash
# Updates the build and revision template to a new git reference

# - Service accounts, roles and secrets must exist in the target
# - kubectl is configured to point to the cluster where the pipeline is executed
# - ~/.wskprops exists and is configured with the correct credentials
# - S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY are defined
# -  ~/.docker/config.json exists and contains credentials for DockerHub

# Optional (with defaults)
GIT_REFERENCE=${GIT_REFERENCE:-$(git rev-parse HEAD)}
GIT_REFERENCE_OLD=${GIT_REFERENCE_OLD:-$(git rev-parse $GIT_REFERENCE~1)}
GIT_URL=${GIT_URL:-https://github.com/mtreinish/ciml}
IMAGES_BASE_URL=${IMAGES_BASE_URL:-index.docker.io/andreaf76}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short HEAD)}
TARGET_NAMESPACE=${TARGET_NAMESPACE:-ciml}
S3_ACCESS_KEY_ID=${S3_ACCESS_KEY_ID:?"Please set S3_ACCESS_KEY_ID"}
S3_SECRET_ACCESS_KEY=${S3_SECRET_ACCESS_KEY:?"Please set S3_SECRET_ACCESS_KEY"}
WSKDEPLOY_CONFIG=${WSKDEPLOY_CONFIG:?"Please set WSKDEPLOY_CONFIG"}
DOCKER_CONFIG=${DOCKER_CONFIG:?"Please set DOCKER_CONFIG"}
SERVICE_ACCOUNT=${SERVICE_ACCOUNT:-default}
TASK_OR_PIPELINE=${TASK_OR_PIPELINE:-pipeline}

BASEDIR=$(ROOT=$(dirname $0); cd $ROOT; pwd)

echo "Building @$GIT_REFERENCE, IMAGE_TAG: $IMAGE_TAG"

## Make sure the namespace exists
kubectl create namespace $TARGET_NAMESPACE 2> /dev/null

## Apply latest task and pipeline definitions
kubectl apply -n $TARGET_NAMESPACE -f ${BASEDIR}/build_and_deploy.yaml

## Create secrets
# S3 credentials for CIML
kubectl delete -n $TARGET_NAMESPACE secret/cimls3credentials &> /dev/null
cat <<EOF | kubectl create -n $TARGET_NAMESPACE -f -
apiVersion: v1
kind: Secret
metadata:
  name: cimls3credentials
type: Opaque
data:
  s3_access_key_id: $(echo $S3_ACCESS_KEY_ID | base64)
  s3_secret_access_key: $(echo $S3_SECRET_ACCESS_KEY | base64)
EOF
# Wskdeploy credentials for deployment
kubectl delete -n $TARGET_NAMESPACE secret/cimlwiskconfig &> /dev/null
kubectl create -n $TARGET_NAMESPACE secret generic cimlwiskconfig --from-file=$WSKDEPLOY_CONFIG
# Docker credentials to deploy OpenWhisk base image
cat <<EOF > /tmp/config.json
{
  "auths": {
    "https://index.docker.io/v1/": $(cat $DOCKER_CONFIG/config.json | jq -c '.auths["https://index.docker.io/v1/"]')
  }
}
EOF
kubectl delete -n $TARGET_NAMESPACE secret/dockerhubconfig &> /dev/null
kubectl create -n $TARGET_NAMESPACE secret generic dockerhubconfig --from-file=.dockerconfigjson=/tmp/config.json --type=kubernetes.io/dockerconfigjson
kubectl get serviceaccount/$SERVICE_ACCOUNT -n $TARGET_NAMESPACE -o json | jq  'del(.metadata.resourceVersion)' |  jq 'del(.secrets[] | select(.name == "dockerhubconfig"))' | jq '.secrets += [{"name": "dockerhubconfig"}]' | kubectl apply -n $TARGET_NAMESPACE -f -

## Setup resources
# GIT
GIT_RESOURCE=$(cat <<EOF | kubectl create -n $TARGET_NAMESPACE -o jsonpath='{.metadata.name}' -f -
apiVersion: tekton.dev/v1alpha1
kind: PipelineResource
metadata:
  generateName: ciml-git-
  labels:
    app: ciml
    tag: "$IMAGE_TAG"
spec:
  type: git
  params:
    - name: revision
      value: $GIT_REFERENCE
    - name: url
      value: $GIT_URL
EOF
)
GIT_RESOURCE_OLD=$(cat <<EOF | kubectl create -n $TARGET_NAMESPACE -o jsonpath='{.metadata.name}' -f -
apiVersion: tekton.dev/v1alpha1
kind: PipelineResource
metadata:
  generateName: ciml-git-
  labels:
    app: ciml
    tag: "$IMAGE_TAG"
spec:
  type: git
  params:
    - name: revision
      value: $GIT_REFERENCE_OLD
    - name: url
      value: $GIT_URL
EOF
)

# Image
IMAGE_RESOURCE=$(cat <<EOF | kubectl create -n $TARGET_NAMESPACE -o jsonpath='{.metadata.name}' -f -
apiVersion: tekton.dev/v1alpha1
kind: PipelineResource
metadata:
  generateName: ciml-action-image-
  labels:
    app: ciml
    tag: "$IMAGE_TAG"
spec:
  type: image
  params:
    - name: url
      value: $IMAGES_BASE_URL/ciml-action-base
EOF
)

if [[ "$TASK_OR_PIPELINE" == "pipeline" ]]; then
  ## Setup and run the pipeline
  cat <<EOF | kubectl create -n $TARGET_NAMESPACE -f -
  apiVersion: tekton.dev/v1alpha1
  kind: PipelineRun
  metadata:
    generateName: ciml-action-build-and-deploy-run-
    labels:
      app: ciml
      tag: "$IMAGE_TAG"
  spec:
    pipelineRef:
      name: ciml-action-build-and-deploy
    params:
      - name: imageTag
        value: $IMAGE_TAG
    serviceAccount: "$SERVICE_ACCOUNT"
    resources:
      - name: src
        resourceRef:
          name: $GIT_RESOURCE
      - name: src-old
        resourceRef:
          name: $GIT_RESOURCE_OLD
      - name: builtImage
        resourceRef:
          name: $IMAGE_RESOURCE
EOF
else
  cat <<EOF | kubectl create -n $TARGET_NAMESPACE -f -
  apiVersion: tekton.dev/v1alpha1
  kind: TaskRun
  metadata:
    generateName: ciml-action-deploy-run-
    labels:
      app: ciml
      tag: "$IMAGE_TAG"
  spec:
    serviceAccount: "$SERVICE_ACCOUNT"
    taskRef:
      name: deploy-actions
    inputs:
      params:
        - name: imageTag
          value: $IMAGE_TAG
      resources:
        - name: workspace
          resourceRef:
            name: $GIT_RESOURCE
EOF
fi

# Watch command
echo "watch kubectl get all -l tag=$IMAGE_TAG -n $TARGET_NAMESPACE"
