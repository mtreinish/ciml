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

import gzip
import itertools
import json
import os
import queue
import re

from ciml import dstat_data
from ciml import gather_results
from ciml import listener
from ciml import svm_trainer

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


default_db_uri = ('mysql+pymysql://query:query@logstash.openstack.org/'
                  'subunit2sql')


def fixed_lenght_example(result, normalized_length=5500):
    """Normalize one example.

    Normalize one example of data to a fixed length (L) and unroll it.
    The input is s x d. To achieve fixed lenght:
    - if s > L, cut each dstat column data to L
    - if s < L, pad with zeros to reach L

    The output is a pd.DataFrame with shape (L, d)
    """
    # Fix length of dataset
    example = result['dstat']
    init_len = len(example)
    dstat_keys = example.keys()

    # Cut or pad with zeros
    if init_len > normalized_length:
        example = example[:normalized_length]
    elif init_len < normalized_length:
        pad_length = normalized_length - init_len
        padd = pd.DataFrame(0, index=np.arange(pad_length), columns=dstat_keys)
        example = pd.concat([example, padd])
    return example


def unroll_example(example, normalized_length=5500, labels=None):
    """Unroll one example and build labels for the unrolled example.

    Unroll one example with shape (L, d) to a pd.Series with shape (L * d,)
    Labels for the input example are an array with shape (d, ), e.g.:
        ['usr', 'sys', ... , 'clo']
    Labels for the output example are an array with shape (L * d, ), e.g.:
        ['usr1', ... , 'usrL', ... , 'clo1', ... , 'cloN']

    Labels are only calculated if not input labels are provide, so we can
    calculate them only once. Labels are a plain python list.

    f = L * d is the number of features for the model.
    """
    # Unroll the examples
    np_vector = example.values.flatten('F')
    if not labels:
        # We need to calculate labels only once (feature names)
        labels = [label + str(idx) for label, idx in itertools.product(
            example.columns, range(normalized_length))]

    vector = pd.Series(np_vector)
    return vector, labels


def get_class(result, class_label='status'):
    """Get a normalized result for the specified class.

    Get a normalized result for the specified class. Currently supported
    classes are only one, 'status'. This returns a single value which
    defines the class the example belongs to.
    """
    if class_label == 'status':
        status = result['status']
        passed_statuses = [0, 'Success']
        status = 0 if status in passed_statuses else 1
        return status
    else:
        return None

def normalize_example(result, normalized_length=5500, labels=None,
                      class_label='status'):
    """Normalize and unroll one example.

    Invokes fixed_lenght_example and unroll_example.
    Returns the unrolled vector, the single integer that represent that status
    for the example, and the list of labels.
    """
    example = fixed_lenght_example(result, normalized_length)
    # Normalize status
    status = get_class(result, class_label)
    vector, labels = unroll_example(example, normalized_length, labels)
    return vector, status, labels


def normalize_dataset(examples, labels, params=None):
    """Normalize features in a dataset

    Normalize each feature in a dataset. If e is the number of examples we have
    in the dataset, and f is the number of features, this takes as input a
    np ndarray with shape (e, f).

    The output is an np ndarray with shape (e, f) where data for each feature
    is normalized based on values across the examples.

    The normalization formula is x = (x - mean(X)/(max(X) - min(X)), where X
    is the vector of feature values across examples, and x is any element of X.
    """
    _features = np.ndarray(shape=(examples.shape[1], examples.shape[0]))
    params = params or {}
    for n in range(len(labels)):
        print("Normalizing feature %d of %d" % (
            n + 1, len(labels)), end='\r', flush=True)
        feature_data = examples[:, n]
        if labels[n] in params:
            mean_fd, max_min_fd = params[labels[n]]
        else:
            mean_fd = np.mean(feature_data)
            max_min_fd = np.max(feature_data) - np.min(feature_data)
            params[labels[n]] = (mean_fd, max_min_fd)
        _features[n] = list(
            map(lambda x: (x - mean_fd) / max_min_fd, feature_data))
    return _features.transpose(), params


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
            event['build_uuid'], 'mqtt-dataset', '1s', default_db_uri)
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


def local_run_uuids(dataset):
    """Return the list of run_uuids available in a local dataset storage"""
    raw_data_folder = os.sep.join([os.path.dirname(os.path.realpath(__file__)),
                                   os.pardir, 'data', dataset, 'raw'])
    run_uuids = [f[:-7] for f in os.listdir(raw_data_folder) if
                 os.path.isfile(os.path.join(raw_data_folder, f)) and
                 f.endswith('.csv.gz')]
    return run_uuids

def get_downsampled_example_lenght(sample_interval, normalized_length=5500):
    """Returns the normalized lenght for a downsampled example

    Returns the normalized example lenght based on the normalized lenght for
    a full sample and the sample interval.
    """
    rng = pd.date_range('1/1/2012', periods=normalized_length, freq='S')
    ts = pd.Series(np.ones(len(rng)), index=rng)
    ts = ts.resample(sample_interval).sum()
    return ts.shape[0]

def save_model_config(dataset, model_config):
    data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                   'data', dataset]
    os.makedirs(os.sep.join(data_folder), exist_ok=True)
    model_config_file = os.sep.join(data_folder + [dataset + '.json.gz'])
    existing_config = load_model_config(dataset)
    # TODO(andreaf) For now we just override things. This would actually be a
    # good place to fail or at least warn users that the model is being
    # re-trained with conflicting parameters.
    if existing_config:
        existing_config.update(model_config)
        model_config = existing_config

    with gzip.open(model_config_file, mode='wb') as local_cache:
        local_cache.write(json.dumps(model_config).encode())

def load_model_config(dataset):
    data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                   'data', dataset]
    model_config_file = os.sep.join(data_folder + [dataset + '.json.gz'])
    if os.path.isfile(model_config_file):
        try:
            with gzip.open(model_config_file, mode='r') as f:
                return json.loads(f.read())
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load this run.
            print('Dataset config found in the local dataset, however: %s', ioe)
            return None

@click.command()
@click.option('--train/--no-train', default=False,
              help="Whether to only build the dataset or train as well.")
@click.option('--estimator', default='tf.estimator.DNNClassifier',
              help='Type of model to be used (not implemented yet).')
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--build-name', default="tempest-full", help="Build name.")
@click.option('--limit', default=0, help="Maximum number of entries")
@click.option('--db-uri', default=default_db_uri, help="DB URI")
def db_trainer(train, estimator, dataset, build_name, limit, db_uri):
    runs = gather_results.get_runs_by_name(db_uri, build_name=build_name)
    model_config = {'build_name': build_name}
    save_model_config(dataset, model_config)
    if limit > 0:
        runs = runs[:limit]
    for run in runs:
        results = gather_results.get_subunit_results_for_run(
            run, dataset, '1s', db_uri)
        print('Acquired run %s' % run.uuid)
        if train:
            train_results(results, dstat_model)


@click.command()
@click.option('--train/--no-train', default=True,
              help="Whether to only build the dataset or train as well.")
@click.option('--estimator', default='tf.estimator.DNNClassifier',
              help='Type of model to be used (not implemented yet).')
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--sample-interval', default=None,
              help='dstat (down)sampling interval')
@click.option('--features-regex', default=None,
              help='List of dstat features to use (column names)')
@click.option('--class-label', default='status',
              help='Label that identifies the type of result for the dataset')
@click.option('--visualize/--no-visualize', default=False,
              help="Visualize data")
@click.option('--steps', default=30, help="Number of training steps")
@click.option('--gpu', default=False, help='Force using gpu')
@click.option('--debug/--no-debug', default=False)
def local_trainer(train, estimator, dataset, sample_interval, features_regex,
                  class_label, visualize, steps, gpu, debug):

    # Our methods expect an object with an uuid field, so build one
    class _run(object):
        def __init__(self, uuid):
            self.uuid = uuid
            self.artifacts = None

    # Normalized lenght before resampling
    normalized_length = 5500
    if sample_interval:
        # Calculate the desired normalized lenght after resample
        normalized_length = get_downsampled_example_lenght(
            sample_interval, normalized_length)

    data_plots_folder = [os.path.dirname(
        os.path.realpath(__file__)), os.pardir, 'data', dataset, 'plots']
    os.makedirs(os.sep.join(data_plots_folder), exist_ok=True)
    run_uuids = local_run_uuids(dataset)

    # run_uuids are the example_ids
    sizes = []
    # The data for each example. We don't know yet the pre-set shape, so
    # wait until the first result comes in
    examples = []

    # Model configuration. We need to cache sample_interval, features-regex and
    # the normalization parameters for each feature so we can re-use them
    # during prediction.
    model_config = {
        'sample_interval': sample_interval,
        'features_regex': features_regex,
        'normalized_length': normalized_length
    }

    # The test result for each example
    classes = []
    labels = []
    idx = 0
    for run in run_uuids:
        results = gather_results.get_subunit_results_for_run(
            _run(run), dataset, sample_interval)
        # For one run_uuid we must only get on example (result)
        result = results[0]
        # Filtering by columns
        df = result['dstat']
        if features_regex:
            col_regex = re.compile(features_regex)
            result['dstat'] = df[list(filter(col_regex.search, df.columns))]
        # Setup the numpy matrix and sizes
        if len(examples) == 0:
            # Adjust normalized_length to the actual re-sample one
            examples = np.ndarray(
                shape=(len(run_uuids),
                       len(result['dstat'].columns) * normalized_length))
            model_config['num_columns'] = len(result['dstat'].columns)
            model_config['num_features'] = (len(
                result['dstat'].columns) * normalized_length)
        # Normalize data
        example = fixed_lenght_example(result, normalized_length)
        # Normalize status
        status = get_class(result, class_label)
        vector, new_labels = unroll_example(example, normalized_length, labels)
        # Only calculate labels for the first example
        if len(labels) == 0:
            labels = new_labels
            model_config['labels'] = labels
        print("Normalized example %d of %d" % (
            run_uuids.index(run) + 1, len(run_uuids)), end='\r', flush=True)
        # Examples is an np ndarrays
        examples[idx] = vector.values
        classes.append(status)
        if visualize:
            # Prepare some more data if we are going to visualize
            sizes.append((result['dstat'].shape[0], status))
            figure_name = sample_interval + "_%s_" + str(idx)
            # Plot un-normalized data
            data_plot = result['dstat'].plot()
            fig = data_plot.get_figure()
            fig.savefig(os.sep.join(
                data_plots_folder + [figure_name % "downsampled"]))
            plt.close(fig)
            # Plot fixed size data
            fixed_plot = example.plot()
            fig = fixed_plot.get_figure()
            fig.savefig(os.sep.join(
                data_plots_folder + [figure_name % "fixedsize"]))
            plt.close(fig)
            # Plot unrolled data
            unrolled_plot = pd.Series(vector).plot()
            fig = unrolled_plot.get_figure()
            fig.savefig(os.sep.join(
                data_plots_folder + [figure_name % "unrolled"]))
            plt.close(fig)
        idx += 1
    # Perform dataset-wise normalization
    # NOTE(andreaf) When we train the model we ignore any saved normalization
    # parameter, since the sample interval and features may be different.
    n_examples, normalization_params = normalize_dataset(examples, labels)
    # We do cache the result to normalize the prediction set.
    model_config['normalization_params'] = normalization_params
    save_model_config(dataset, model_config)
    if visualize:
        for n in range(len(run_uuids)):
            figure_name = sample_interval + "_%s_" + str(n)
            unrolled_norm_plot = pd.Series(n_examples[n]).plot()
            fig = unrolled_norm_plot.get_figure()
            fig.savefig(os.sep.join(
                data_plots_folder + [figure_name % "normalized"]))
            plt.close(fig)

        np_sizes = np.array(sizes)
        df = pd.DataFrame(np_sizes, columns=['size', 'status'])
        size_plot = df.plot.scatter(x='size', y='status')
        fig = size_plot.get_figure()
        fig.savefig(os.sep.join(data_plots_folder + ['sizes_by_result.png']))
        plt.close(fig)

    # Now do the training
    classes = np.array(classes)
    print("\nTraining data shape: (%d, %d)" % n_examples.shape)
    if train:
        if debug:
            tf.logging.set_verbosity(tf.logging.DEBUG)
        config = tf.ConfigProto(log_device_placement=True,)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        model = svm_trainer.SVMTrainer(n_examples, run_uuids, labels,
                                       classes, dataset_name=dataset,
                                       force_gpu=gpu)
        model.train(steps=steps)
