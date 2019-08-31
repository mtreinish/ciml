FROM ubuntu
RUN apt-get update && apt-get -y install curl
RUN curl -L 'https://github.com/apache/openwhisk-wskdeploy/releases/download/latest/openwhisk_wskdeploy-latest-linux-amd64.tgz' | tar zxvf - -C /usr/local/bin wskdeploy
