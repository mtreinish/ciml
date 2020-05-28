FROM alpine:3.11
LABEL maintainer "Andrea Frittoli <andrea.frittoli@gmail.com>"

ARG TKN_VERSION=0.9.0

RUN wget -O- https://github.com/tektoncd/cli/releases/download/v${TKN_VERSION}/tkn_${TKN_VERSION}_Linux_x86_64.tar.gz | tar zxf - -C /usr/local/bin