FROM alpine:3.10
LABEL maintainer "Andrea Frittoli <andrea.frittoli@gmail.com>"

RUN apk update; apk upgrade; apk add bash util-linux

ARG KUBECTL_VERSION=1.16.0
RUN wget -O/usr/local/bin/kubectl https://storage.googleapis.com/kubernetes-release/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl; chmod +x /usr/local/bin/kubectl

ARG TKN_VERSION=0.5.0
RUN wget -O- https://github.com/tektoncd/cli/releases/download/v${TKN_VERSION}/tkn_${TKN_VERSION}_Linux_x86_64.tar.gz | tar zxf - -C /usr/local/bin
