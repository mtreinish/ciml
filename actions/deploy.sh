#!/bin/bash

##############################################################################
# Copyright 2018 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

# Define useful folders
actions_folder=$(cd $(dirname $0); pwd)
root_folder=$(cd ${actions_folder}/..; pwd)

function usage() {
  echo -e "Usage: $0 [--install,--uninstall,--env] [extra wskdeploy params]"
}

function install() {
  shift
  # Copy the gather_results module around
  cp ${root_folder}/ciml/gather_results.py ${actions_folder}/cache-data/__main__.py
  # Install
  wskdeploy -p ${actions_folder}/ $@
}

function uninstall() {
  shift
  # Uninstall
  wskdeploy undeploy -p ${actions_folder}/ $@
}

case "$1" in
"--install" )
install
;;
"--uninstall" )
uninstall
;;
"--env" )
env
;;
* )
usage
;;
esac
