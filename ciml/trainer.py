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

import os
import queue

from ciml import dstat_data
from ciml import gather_results
from ciml import listener
from ciml import svm_trainer

import click
import numpy as np
import pandas as pd


default_db_uri = ('mysql+pymysql://query:query@logstash.openstack.org/'
                  'subunit2sql')


def normalize_example(result, normalized_length=5500):
    # Normalize one sample of data. This is a dstat file which is turned
    # into a fixed lenght vetor.

    # Fix length of dataset
    init_len = len(result)
    example = result['dstat']
    dstat_keys = example.keys()

    if init_len > normalized_length:
        example = example[:normalized_length]
    elif init_len < normalized_length:
        pad_length = normalized_length - init_len
        padd = pd.DataFrame(0, index=np.arange(pad_length), columns=dstat_keys)
        example = pd.concat([example, padd])
    # Vectorize the dataframe
    vector = pd.Series()
    labels = pd.Series()
    # Vectors are unrolled examples
    # Labels are example_ids
    for key in dstat_keys:
        vector = pd.concat([vector, example[key]])
        labels = pd.concat(
            [labels, pd.Series(key, index=np.arange(normalized_length))])
    # Status is 0 for success, 1 for fail
    status = 0 if result['status'] == 'Success' else 1
    return vector, labels, status


def train_results(results, model):
    for result in results:
        # Normalize data - take the whole data as we may need results
        # for an effective normalization
        vector, labels, status = normalize_example(result)
        # Do not train just yet
        model.train(vector, status)


def mqtt_trainer():
    event_queue = queue.Queue()
    listen_thread = listener.MQTTSubscribe(event_queue,
                                           'firehose.openstack.org',
                                           'gearman-subunit/#')
    listen_thread.start()
#    dstat_model = dstat_data.DstatTrainer('mqtt-dataset')
    while True:
        event = event_queue.get()
        results = gather_results.get_subunit_results(
            event['build_uuid'], 'mqtt-dataset', default_db_uri)
        examples = []
        classes = []
        labels_list = []
        for result in results:
            vector, labels, status = normalize_example(result)
            examples = vector
            classes = status
            labels_list = labels
            if vector and labels and status:
                break
        run_uuids = [event['build_uuid']] * len(examples)
        dstat_model = svm_trainer.SVMTrainer(examples, run_uuids, labels_list,
                                             classes)
        dstat_model.train()


@click.command()
@click.option('--train/--no-train', default=False,
              help="Whether to only build the dataset or train as well.")
@click.option('--estimator', default='tf.estimator.DNNClassifier',
              help='Type of model to be used (not implemented yet).')
@click.option('--dataset', default="dataset",
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
@click.option('--train/--no-train', default=True,
              help="Whether to only build the dataset or train as well.")
@click.option('--estimator', default='tf.estimator.DNNClassifier',
              help='Type of model to be used (not implemented yet).')
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--visualize/--no-visualize', default=False,
              help="Visualize data")
def local_trainer(train, estimator, dataset, visualize):

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
    # run_uuids are the example_ids
    sizes = []
    # The data for each example
    examples = []
    # The test result for each example
    classes = []
    for run in run_uuids:
        results = gather_results.get_subunit_results_for_run(
            _run(run), dataset)
        # For one run_uuid we must only get on example (result)
        result = results[0]
        vector, labels, status = normalize_example(result)
        examples.append(vector)
        classes.append(status)
        if visualize:
            # Prepare some more data if we are going to visualize
            sizes.append((result['dstat'].shape[0], status))

    if visualize:
        np_sizes = np.array(sizes)
        df = pd.DataFrame(np_sizes, columns=['size', 'status'])
        size_plot = df.plot.scatter(x='size', y='status')
        fig = size_plot.get_figure()
        fig.savefig(raw_data_folder + '/sizes_by_result.png')

    # Now do the training
    # TODO(andreaf) We should really train on half of the examples
    # The cross-validate on the other half
    # And finally predict on the MQTT inputs
    if train:
        model = svm_trainer.SVMTrainer(examples, run_uuids, labels, classes)
        model.train()
