# Copyright 2018 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import queue
import os

from ciml import dstat_data
from ciml import gather_results
from ciml import listener

import click

default_db_uri = 'mysql+pymysql://query:query@logstash.openstack.org/subunit2sql'

def normalize_data(result):
    # Normalize data. This is key to a good prediction.
    # no-op for now
    return result


def train_results(results, model):
    for result in results:
        # Normalize data - take the whole data as we may need results
        # for an effective normalization
        nresult = normalize_data(result)
        # Do not train just yet
        # model.train(nresult['dstat'], nresult['status'])


def mqtt_trainer():
    event_queue = queue.Queue()
    listen_thread = listener.MQTTSubscribe(event_queue,
                                           'firehose.openstack.org',
                                           'gearman-subunit/#')
    listen_thread.start()
    dstat_model = dstat_data.DstatTrainer('mqtt-dataset')
    while True:
        event = event_queue.get()
        results = gather_results.get_subunit_results(
            event['build_uuid'], 'mqtt-dataset', db_uri)
        train_results(results, dstat_model)


@click.command()
@click.option('--train/--no-train', default=False,
              help="Whether to only build the dataset or train as well.")
@click.option('--estimator', default='tf.estimator.DNNClassifier',
              help='Type of model to be used (not implemented yet).')
@click.option('--dataset',  default="dataset",
              help="Name of the dataset folder.")
@click.option('--build-name', default="tempest-full", help="Build name.")
@click.option('--db-uri', default=default_db_uri, help="DB URI")
def db_trainer(train, estimator, dataset, build_name, db_uri):
    runs = gather_results.get_runs_by_name(db_uri, build_name=build_name)
    if train:
        dstat_model = dstat_data.DstatTrainer(dataset)
    for run in runs:
        results = gather_results.get_subunit_results_for_run(
            run, dataset, db_uri)
        if train:
            train_results(results, dstat_model)


@click.command()
@click.option('--estimator', default='tf.estimator.DNNClassifier',
              help='Type of model to be used (not implemented yet).')
@click.option('--dataset',  default="dataset",
              help="Name of the dataset folder.")
def local_trainer(estimator, dataset):

    # Our methods expect an object with an uuid field, so build one
    class _run(object):
        def __init__(self, uuid):
            self.uuid = uuid
            self.artifacts = None

    raw_data_folder = os.sep.join([os.path.dirname(os.path.realpath(__file__)),
                                   os.pardir, 'data', dataset, 'raw'])
    run_uuids = [f[:-7] for f in os.listdir(raw_data_folder) if
                 os.path.isfile(os.path.join(raw_data_folder, f)) and
                 f.endswith('.csv.gz')]
    dstat_model = dstat_data.DstatTrainer(dataset)
    for run in run_uuids:
        results = gather_results.get_subunit_results_for_run(
            _run(run), dataset)
        train_results(results, dstat_model)
