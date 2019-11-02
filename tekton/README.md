# Create a dataset using Tekton

This Task allows running CIML in a Tekton task that takes the raw dataset
from an s3 bucket and creates a normalised dataset in another s3 bucket.

To use the task:
* Define the TARGET_NAMESPACE and install the task
```
# Setup KUBECONFIG or kubectl config use-context [your-context]
export TARGET_NAMESPACE=${TARGET_NAMESPACE:-default}
kubectl apply -f tekton/ -n $TARGET_NAMESPACE
```

* Create the pipeline resource for the ciml git:
```
GIT_RESOURCE=$(cat <<EOF | kubectl create -n $TARGET_NAMESPACE -o jsonpath='{.metadata.name}' -f -
apiVersion: tekton.dev/v1alpha1
kind: PipelineResource
metadata:
  generateName: ciml-git-
  labels:
    app: ciml
spec:
  type: git
  params:
    - name: revision
      value: master
    - name: url
      value: https://github.com/mtreinish/ciml
EOF
)
```

* Create the secret that olds the .aws configuration files
```
kubectl create secret generic s3storage --from-literal aws_access_key_id=$AWS_ACCESS_KEY_ID --from-literal aws_secret_access_key=$AWS_SECRET_ACCESS_KEY -n $TARGET_NAMESPACE
```

* Run the task using a taskrun
```
cat <<EOF | kubectl create -n $TARGET_NAMESPACE -f -
apiVersion: tekton.dev/v1alpha1
kind: TaskRun
metadata:
  generateName: ciml-create-dataset-
spec:
  inputs:
    params:
    - name: dataset
      value: size-mean-std-used-status
    - name: aggregation-functions
      value: "mean size std"
    resources:
    - name: ciml
      resourceRef:
        name: $GIT_RESOURCE
  taskRef:
    name: ciml-create-dataset
EOF
```
