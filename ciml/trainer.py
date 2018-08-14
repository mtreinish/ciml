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

import itertools
import os
import queue
import re
import sys
import warnings
warnings.filterwarnings("ignore")


from ciml import gather_results
from ciml import listener
from ciml import nn_trainer
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


def unroll_example(example, normalized_length=5500):
    """Unroll one example

    Unroll one example with shape (L, d) to a pd.Series with shape (L * d,)
    Labels for the input example are an array with shape (d, ), e.g.:
        ['usr', 'sys', ... , 'clo']
    Labels for the output example are an array with shape (L * d, ), e.g.:
        ['usr1', ... , 'usrL', ... , 'clo1', ... , 'cloN']

    f = L * d is the number of features for the model.
    """
    # Unroll the examples
    np_vector = example.values.flatten('F')
    return pd.Series(np_vector)


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


def normalize_example(result, normalized_length=5500, class_label='status'):
    """Normalize and unroll one example.

    Invokes fixed_lenght_example and unroll_example.
    Returns the unrolled vector, the single integer that represent that status
    for the example, and the list of labels.
    """
    example = fixed_lenght_example(result, normalized_length)
    # Normalize status
    status = get_class(result, class_label)
    vector = unroll_example(example, normalized_length)
    return vector, status


def filter_example(result, features_regex):
    """Filters the dstat data by features_regex"""
    # Apply the dstat feature filter
    dstat_data = result['dstat']
    col_regex = re.compile(features_regex)
    result['dstat'] = dstat_data[list(filter(
        col_regex.search, dstat_data.columns))]
    return result


def unroll_labels(dstat_labels, normalized_length=5500):
    """Build labels for the unrolled example from the list of runs"""
    # Get one run
    return [label + str(idx) for label, idx in itertools.product(
        dstat_labels, range(normalized_length))]


def examples_ndarray(num_examples, num_dstat_features, normalized_length):
    # Setup the numpy matrix and sizes (this is done once)
    return np.ndarray(
        shape=(num_examples, num_dstat_features * normalized_length))


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
    print(flush=True)
    return _features.transpose(), params


def get_downsampled_example_lenght(sample_interval, normalized_length=5500):
    """Returns the normalized lenght for a downsampled example

    Returns the normalized example lenght based on the normalized lenght for
    a full sample and the sample interval.
    """
    rng = pd.date_range('1/1/2012', periods=normalized_length, freq='S')
    ts = pd.Series(np.ones(len(rng)), index=rng)
    ts = ts.resample(sample_interval).sum()
    return ts.shape[0]


def data_sizes_and_labels(sample_run, features_regex, sample_interval='1s'):
    """Takes a sample run from a dataset and filters and does calculations

    Returns:
    - the normalized example lenght
    - the number of dstat features
    - the unrolled labels
    """
    # Normalized lenght before resampling
    normalized_length = 5500
    if sample_interval:
        # Calculate the desired normalized lenght after resample
        normalized_length = get_downsampled_example_lenght(
            sample_interval, normalized_length)

    # Load the list of runs and base labels
    sample_result = gather_results.get_subunit_results_for_run(
        sample_run, sample_interval)
    filtered_sample_result = filter_example(sample_result, features_regex)
    filtered_dstat_labels = filtered_sample_result['dstat'].columns
    unrolled_labels = unroll_labels(filtered_dstat_labels, normalized_length)
    return normalized_length, len(filtered_dstat_labels), unrolled_labels


def prepare_dataset(dataset, normalized_length, num_dstat_features, data_type,
                    features_regex, sample_interval='1s',
                    class_label='status', visualize=False):
    """Takes a dataset and filters and does the magic

    Loads the run ids from the dataset configuration.
    Loads the data (dsv + meta) for every run from cache.
    Builds the unrolled exaples as a numpy ndarray.
    Builds the classes as a numpy array.
    Saves the data setup to the dataset config.
    Does some visualization (if enabled).
    """
    if visualize:
        data_plots_folder = [os.path.dirname(
            os.path.realpath(__file__)), os.pardir, 'data', dataset, 'plots']
        os.makedirs(os.sep.join(data_plots_folder), exist_ok=True)

    # Load the list of runs and base labels
    runs = gather_results.load_run_uuids(dataset, name=data_type)

    # run_uuids are the example_ids
    sizes = []
    # The data for each example.
    examples = examples_ndarray(len(runs), num_dstat_features,
                                normalized_length)

    # The test result for each example
    classes = []
    skips = []
    print("Loading %s data:" % data_type, end='\r', flush=True)
    for count, run in enumerate(runs):
        print("Loading %s data: %d of %d" % (data_type, count + 1, len(runs)),
              end='\r', flush=True)
        result = gather_results.get_subunit_results_for_run(
            run, sample_interval)
        # For one run_uuid we must only get on example (result)
        # Filtering by columns
        if not result:
            skips.append(run.uuid)
            continue

        # Apply column filtering
        result = filter_example(result, features_regex)

        # Normalize data
        example = fixed_lenght_example(result, normalized_length)
        vector = unroll_example(example, normalized_length)

        # Normalize status
        status = get_class(result, class_label)

        # Examples is an np ndarrays
        examples[count] = vector.values
        classes.append(status)

        # Plot from figures
        if visualize:
            # Prepare some more data if we are going to visualize
            sizes.append((result['dstat'].shape[0], status))
            figure_name = sample_interval + "_%s_" + str(count)
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

    print("Loading %s data: %d done!" % (data_type, len(runs)))
    # Check that everything went well
    if len(skips) > 0:
        print('Unable to train model because of missing runs %s' % skips)
        safe_runs = [run.uuid for run in runs if run.uuid not in skips]
        gather_results.save_run_uuids(dataset, safe_runs)
        print('The model has been updated to exclude those runs.')
        print('Please re-run the training step.')
        sys.exit(1)

    classes = np.array(classes)
    figure_sizes = np.array(sizes)
    example_ids = np.array(runs)

    print("%s set: examples: %s, classes: %s, example IDs: %s" % (
        data_type, str(examples.shape), str(classes.shape),
        str(example_ids.shape)))

    data = {
        'examples': examples,
        'example_ids': example_ids,
        'classes': classes
    }

    return data, figure_sizes

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
            vector, status = normalize_example(result)
            examples = vector
            classes = status
            labels_list = labels
            if vector and labels and status:
                break
        run_uuids = [event['build_uuid']] * len(examples)
        dstat_model = svm_trainer.SVMTrainer(examples, run_uuids, labels_list,
                                             classes)
        dstat_model.train()

def dataset_split_filters(size, training, dev):
    # Separate training, dev and test sets by building randomised lists
    # of indexes that can be used for examples, example_ids and classes
    all_indexes = range(size)
    training_idx = pd.Series(all_indexes).sample(int(size * training)).values
    non_training_idx = list(set(all_indexes) - set(training_idx))
    remaining_size = int(size * (1 - training))
    dev_idx = pd.Series(non_training_idx).sample(
        int(remaining_size * dev)).values
    test_idx = list(set(non_training_idx) - set(dev_idx))
    return training_idx, dev_idx, test_idx

@click.command()
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--build-name', default="tempest-full", help="Build name.")
@click.option('--limit', default=0, help="Maximum number of entries")
@click.option('--sample-interval', default=None,
              help='dstat (down)sampling interval')
@click.option('--features-regex', default=None,
              help='List of dstat features to use (column names)')
@click.option('--class-label', default='status',
              help='Label that identifies the type of result for the dataset')
@click.option('--tdt-split', nargs=3, type=int, default=(6, 2, 2),
              help='Trainig, dev and test dataset split - sum to 10')
@click.option('--visualize/--no-visualize', default=False,
              help="Visualize data")
def build_dataset(dataset, build_name, limit, sample_interval, features_regex,
                  class_label, tdt_split, visualize):
    # Prevent overwrite by mistake
    if gather_results.load_model_config(dataset):
        print("Dataset %s already configured" % dataset)
        sys.exit(1)

    # Validate tdt-split
    training, dev, test = map(lambda x: x/10, tdt_split)
    if not (training + dev + test) == 1:
        print("Training (%d) + dev (%d) + test (%d) != 10" % tdt_split)
        sys.exit(1)

    # Load available run ids for the build name
    runs = gather_results.load_run_uuids('.raw', name=build_name)
    # Apply the limit
    if limit > 0:
        runs = np.array(runs[:limit])
    print("Obtained %d runs for build %s" % (len(runs), build_name))

    # Split the runs in training, dev and test
    training_idx, dev_idx, test_idx = dataset_split_filters(
        len(runs), training, dev)
    np_runs = np.array(runs)
    gather_results.save_run_uuids(dataset, np_runs[training_idx],
                                  name='training')
    gather_results.save_run_uuids(dataset, np_runs[dev_idx], name='dev')
    gather_results.save_run_uuids(dataset, np_runs[test_idx], name='test')

    # Calculate normalized and filtered dimensions and labels
    normalized_length, num_dstat_features, labels = \
        data_sizes_and_labels(runs[0], features_regex, sample_interval)

    model_config = {
        'build_name': build_name,
        'sample_interval': sample_interval,
        'features_regex': features_regex,
        'class_label': class_label,
        'training_set': training,
        'dev_set': dev,
        'test_set': test,
        'normalized_length': normalized_length,
        'labels': labels,
        'num_columns': num_dstat_features,
        'num_features': len(labels)
    }

    # Save the config and complete list of run uuids
    gather_results.save_run_uuids(dataset, runs)
    gather_results.save_model_config(dataset, model_config)
    print("Stored %d run IDs in the model %s config" % (len(runs), dataset))

    datasets = {}
    # Training must come first so we calculate normalization params
    for data_type in ['training', 'dev', 'test']:
        data, _figure_sizes = prepare_dataset(
            dataset, normalized_length, num_dstat_features, data_type,
            features_regex=features_regex,
            sample_interval=sample_interval, class_label=class_label,
            visualize=visualize)
        datasets[data_type] = data
        examples = data['examples']

        # Perform dataset-wise normalization
        if data_type == 'training':
            n_examples, normalization_params = normalize_dataset(
                examples, labels)
            # We cache normalization parameters from the training data set
            # to normalize the dev and test set, as well as other input data
            model_config['normalization_params'] = normalization_params
            gather_results.save_model_config(dataset, model_config)

            # Save figure sizes as well for training only
            figure_sizes = _figure_sizes
        else:
            # Perform dataset-wise normalization
            n_examples, normalization_params = normalize_dataset(
                examples, labels, model_config['normalization_params'])

        # Replace examples with normalized ones
        datasets[data_type]['examples'] = n_examples

        # Store the normalized data to disk
        gather_results.save_dataset(dataset, name=data_type,
                                    **datasets[data_type])


    # Plot some more figures
    if visualize:
        for n in range(n_examples.shape(1)):
            figure_name = sample_interval + "_%s_" + str(n)
            unrolled_norm_plot = pd.Series(n_examples[n]).plot()
            fig = unrolled_norm_plot.get_figure()
            fig.savefig(os.sep.join(
                data_plots_folder + [figure_name % "normalized"]))
            plt.close(fig)

        df = pd.DataFrame(figure_sizes, columns=['size', 'status'])
        size_plot = df.plot.scatter(x='size', y='status')
        fig = size_plot.get_figure()
        fig.savefig(os.sep.join(data_plots_folder + ['sizes_by_result.png']))
        plt.close(fig)

    # Store labels to disk
    gather_results.save_dataset(dataset, name='labels', labels=labels)


@click.command()
@click.option('--estimator', default='tf.estimator.DNNClassifier',
              help='Type of model to be used (not implemented yet).')
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--steps', default=30, help="Number of training steps")
@click.option('--gpu', default=False, help='Force using gpu')
@click.option('--debug/--no-debug', default=False)
def local_trainer(estimator, dataset, steps, gpu, debug):
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    config = tf.ConfigProto(log_device_placement=True,)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # Load the normalized data
    labels = gather_results.load_dataset(dataset, 'labels')
    training_data = gather_results.load_dataset(dataset, 'training')
    test_data = gather_results.load_dataset(dataset, 'test')
    print("\nTraining data shape: (%d, %d)" % training_data['examples'].shape)
    print("\nEvaluation data shape: (%d, %d)" % test_data['examples'].shape)
    model = svm_trainer.SVMTrainer(labels, training_data, test_data,
                                   dataset_name=dataset, force_gpu=gpu)
    model.train(steps=steps)
