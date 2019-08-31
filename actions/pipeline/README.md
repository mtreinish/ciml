# Actions pipeline

This folder includes a [Tekton](https://github.com/tektoncd/pipeline) pipeline to
build a new version of the OpenWhisk Action base image, and to redeploy the
OpenWhisk actions with the updated image tag.

# Configuration

The service account used for image build should be [configured](https://github.com/tektoncd/pipeline/blob/master/docs/auth.md#basic-authentication-docker) with the docker
credentials required to push the image.
The deployment task expects a secret to hold the .wskprops file used by wskdeploy.
