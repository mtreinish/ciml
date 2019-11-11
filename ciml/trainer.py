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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as cmx
import click
from ciml import listener
from ciml import gather_results
import datetime
import itertools
import os
import queue
import re
import sys
import warnings
warnings.filterwarnings("ignore")


try:
    from ciml import nn_trainer
    from ciml import svm_trainer
    from ciml import tf_trainer
except ImportError:
    print("Warning: could not import CIML trainers")

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

try:
    import tensorflow as tf
    from tensorflow.python.training import adagrad
    from tensorflow.python.training import adam
    from tensorflow.python.training import ftrl
    from tensorflow.python.training import gradient_descent
    from tensorflow.python.training import rmsprop
    from tensorflow.python.training import proximal_adagrad
    _OPTIMIZER_CLS_NAMES = {
        'Adagrad': adagrad.AdagradOptimizer,
        'Adam': adam.AdamOptimizer,
        'Ftrl': ftrl.FtrlOptimizer,
        'RMSProp': rmsprop.RMSPropOptimizer,
        'SGD': gradient_descent.GradientDescentOptimizer,
        'ProximalAdagrad': proximal_adagrad.ProximalAdagradOptimizer
    }
except ImportError:
    print("Warning: could not import Tensorflow")
    _OPTIMIZER_CLS_NAMES = {}

default_db_uri = ('mysql+pymysql://query:query@logstash.openstack.org/'
                  'subunit2sql')


def fixed_lenght_example(result, normalized_length=5500,
                         aggregation_functions=None):
    """Normalize one example.

    Normalize one example of data to a fixed length (L).
    The input is s x d.
    To achieve fixed lenght:
    - if aggregation functions are provided, apply them, or else
    - if s > L, cut each dstat column data to L
    - if s < L, pad with zeros to reach L

    The output is a pd.DataFrame with shape (L, d)
    """
    # Fix length of dataset
    example = result['dstat']
    init_len = len(example)
    dstat_keys = example.keys()

    if aggregation_functions:
        # Run all aggregation functions on each DataFrame column in the example
        agg_dict = {column: [x(example[column]) for x in aggregation_functions]
                    for column
                    in example.columns}
        example = pd.DataFrame.from_dict(agg_dict)
    else:
        # Cut or pad with zeros
        if init_len > normalized_length:
            example = example[:normalized_length]
        elif init_len < normalized_length:
            pad_length = normalized_length - init_len
            padd = pd.DataFrame(0, index=np.arange(
                pad_length), columns=dstat_keys)
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
    elif class_label == 'node_provider':
        provider = result['node_provider']
        if provider.startswith('rax'):
            return 'rax'
        elif provider.startswith('ovh'):
            return 'ovh'
        elif provider.startswith('vexxhost'):
            return 'vexxhost'
        else:
            return provider
    elif class_label == 'node_provider_all':
        return result['node_provider']
    else:
        return result[class_label]


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
    """Build labels for the unrolled example from labels and num of samples"""
    return [label + str(idx) for label, idx in itertools.product(
        dstat_labels, range(normalized_length))]


def unroll_labels_names(dstat_labels, aggregation_functions):
    """Build labels for the unrolled example from lables and agg fns"""
    return [label + '_' + fn for label, fn in itertools.product(
        dstat_labels, aggregation_functions)]


def examples_ndarray(num_examples, num_dstat_features, normalized_length):
    # Setup the numpy matrix and sizes (this is done once)
    return np.ndarray(
        shape=(num_examples, num_dstat_features * normalized_length))


def normalize_dataset(examples, labels, params=None):
    """Normalize features in a dataset

    Normalize each feature in a dataset. If e is the number of examples we have
    in the dataset, and f is the number of features, this takes as input an
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
            # In case of just one example, or
            if max_min_fd == 0:
                max_min_fd = 1
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


def data_sizes_and_labels(sample_run, features_regex, sample_interval='1s',
                          aggregation_functions=None, data_path=None, s3=None):
    """Takes a sample run from a dataset and filters and does calculations

    If aggregation functions are used, the number of features is the
    number of aggregation functions x the number of dstat features
    Returns:
    - the normalized example lenght
    - the number of dstat features
    - the unrolled labels
    """
    # Normalized lenght before resampling
    normalized_length = 5500
    if aggregation_functions:
        normalized_length = len(aggregation_functions)
    elif sample_interval:
        # Calculate the desired normalized lenght after resample
        normalized_length = get_downsampled_example_lenght(
            sample_interval, normalized_length)

    # Load the list of runs and base labels
    sample_result = gather_results.get_subunit_results_for_run(
        sample_run, sample_interval, data_path=data_path, s3=s3)
    filtered_sample_result = filter_example(sample_result, features_regex)
    filtered_dstat_labels = filtered_sample_result['dstat'].columns
    if aggregation_functions:
        unrolled_labels = unroll_labels_names(filtered_dstat_labels,
                                              aggregation_functions)
    else:
        unrolled_labels = unroll_labels(filtered_dstat_labels,
                                        normalized_length)
    return normalized_length, len(filtered_dstat_labels), unrolled_labels


def prepare_dataset(dataset, normalized_length, num_dstat_features, data_type,
                    features_regex, sample_interval='1s', class_label='status',
                    aggregation_functions=None, visualize=False, data_path=None,
                    target_data_path=None, s3=None):
    """Takes a dataset and filters and does the magic

    Loads the run ids from the dataset configuration.
    Loads the data (dsv + meta) for every run from cache.
    Builds the unrolled examples as a numpy ndarray.
    Builds the classes as a numpy array.
    Saves the data setup to the dataset config.
    Does some visualization (if enabled).
    """
    if visualize:
        data_plots_folder = [os.path.dirname(
            os.path.realpath(__file__)), os.pardir, 'data', dataset, 'plots']
        os.makedirs(os.sep.join(data_plots_folder), exist_ok=True)

    # Load the list of runs and base labels
    runs = gather_results.load_run_uuids(dataset, name=data_type,
                                         data_path=target_data_path, s3=s3)

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
            run, sample_interval, data_path=data_path, s3=s3)
        # For one run_uuid we must only get on example (result)
        # Filtering by columns
        if not result:
            skips.append(run.uuid)
            continue

        # Apply column filtering
        result = filter_example(result, features_regex)

        # Normalize data
        example = fixed_lenght_example(result, normalized_length,
                                       aggregation_functions)

        vector = unroll_example(example, normalized_length)

        # Normalize status
        status = get_class(result, class_label)

        # Examples is an np ndarrays
        examples[count] = vector.values
        classes.append(status)

        # Plot from figures
        if visualize and not aggregation_functions:
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

    if visualize and aggregation_functions and len(examples) > 0:
        if len(aggregation_functions) > 3:
            print('Visualization skipped, cannot represent more than 3D')
            sys.exit(1)
        else:
            fig = plt.figure()
            if len(aggregation_functions) == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

            # Build a dict [class] -> [int ID]
            unique_classes = list(set(classes))
            dict_classes = dict(zip(unique_classes,
                                    list(range(len(unique_classes)))))

            # Setup colours
            cm = plt.get_cmap('jet')
            cNorm = pltcolors.Normalize(
                vmin=0, vmax=len(unique_classes))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

            # Scatter the data
            for ii in range(len(examples)):
                ax.scatter(*examples[ii], marker='o',
                           c=scalarMap.to_rgba(dict_classes[classes[ii]]))

            # Set axis labels
            ax.set_xlabel(aggregation_functions[0].__name__)
            if len(aggregation_functions) > 1:
                ax.set_ylabel(aggregation_functions[1].__name__)
            if len(aggregation_functions) > 2:
                ax.set_zlabel(aggregation_functions[2].__name__)

            # scalarMap.set_array(classes)
            # fig.colorbar(scalarMap)

            # Save the plot
            fig.savefig(os.sep.join(
                data_plots_folder + [data_type + "_3d_plot"]))
            plt.close(fig)

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
    training_idx = []
    dev_idx = []
    test_idx = []
    if training > 0:
        training_idx = pd.Series(all_indexes).sample(
            int(size * training)).values
    non_training_idx = list(set(all_indexes) - set(training_idx))
    if dev > 0:
        dev_idx = pd.Series(non_training_idx).sample(int(size * dev)).values
    if (training + dev) < 1:
        test_idx = list(set(non_training_idx) - set(dev_idx))
    return training_idx, dev_idx, test_idx


def resolve_aggregation_function(function_name):
    # Resolve an aggregation function name to a function
    # Searches first numpy, then this module
    if hasattr(np, function_name):
        return getattr(np, function_name)
    elif hasattr(sys.modules[__name__], function_name):
        return getattr(sys.modules[__name__], function_name)
    else:
        raise NameError("Could not find function name %s in 'numpy' or %s",
                        function_name, sys.modules[__name__])


@click.command()
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--build-name', default="tempest-full", help="Build name.")
@click.option('--slicer', default=":", help="Slice of the dataset")
@click.option('--sample-interval', default=None,
              help='dstat (down)sampling interval')
@click.option('--features-regex', default=None,
              help='List of dstat features to use (column names)')
@click.option('--class-label', default='status',
              help='Label that identifies the type of result for the dataset')
@click.option('--tdt-split', nargs=3, type=int, default=(6, 2, 2),
              help='Trainig, dev and test dataset split - sum to 10')
@click.option('--force/--no-force', default=False,
              help='When True, override existing dataset config')
@click.option('--visualize/--no-visualize', default=False,
              help="Visualize data")
@click.option('--data-path', default=None,
              help="Path to the raw data, local path or s3://<bucket>")
@click.option('--target-data-path', default=None,
              help="Path where to store data, local path or s3://<bucket>")
@click.option('--s3-profile', default=None, help='Named configuration')
@click.option('--s3-url',
              default='https://s3.eu-geo.objectstorage.softlayer.net',
              help='Endpoint URL for the s3 storage')
@click.option('--data-plots-folder', default="/tmp",
              help="Folder where plots are stored")
@click.option('--aggregation-functions', default=None, multiple=True,
              help="List of aggregation functions to apply to each sample")
def build_dataset(dataset, build_name, slicer, sample_interval, features_regex,
                  class_label, tdt_split, force, visualize, data_path,
                  target_data_path, s3_profile, s3_url, data_plots_folder,
                  aggregation_functions):
    # s3 support
    s3 = gather_results.get_s3_client(s3_url=s3_url, s3_profile=s3_profile)

    # Prevent overwrite by mistake
    if gather_results.load_model_config(dataset, data_path=target_data_path,
                                        s3=s3) and not force:
        print("Dataset %s already configured" % dataset)
        sys.exit(1)

    # Validate tdt-split
    training, dev, test = map(lambda x: x / 10, tdt_split)
    if not (training + dev + test) == 1:
        print("Training (%d) + dev (%d) + test (%d) != 10" % tdt_split)
        sys.exit(1)

    # Load available run ids for the build name (from s3)
    runs = gather_results.load_run_uuids('.raw', name=build_name,
                                         data_path=data_path, s3=s3)
    # Apply the slice
    def slice_fn(x): return int(x.strip()) if x.strip() else None
    slice_object = slice(*map(slice_fn, slicer.split(":")))
    runs = np.array(runs[slice_object])
    print("Obtained %d runs for build %s" % (len(runs), build_name))

    # Split the runs in training, dev and test
    training_idx, dev_idx, test_idx = dataset_split_filters(
        len(runs), training, dev)
    np_runs = np.array(runs)
    # Saving dataset metadata
    gather_results.save_run_uuids(dataset, np_runs[training_idx],
                                  name='training', data_path=target_data_path,
                                  s3=s3)
    gather_results.save_run_uuids(dataset, np_runs[dev_idx], name='dev',
                                  data_path=target_data_path, s3=s3)
    gather_results.save_run_uuids(dataset, np_runs[test_idx], name='test',
                                  data_path=target_data_path, s3=s3)

    # Calculate normalized and filtered dimensions and labels
    normalized_length, num_dstat_features, labels = \
        data_sizes_and_labels(runs[0], features_regex, sample_interval,
                              aggregation_functions=aggregation_functions,
                              data_path=data_path, s3=s3)
    model_config = {
        'build_name': build_name,
        'sample_interval': sample_interval,
        'features_regex': features_regex,
        'class_label': class_label,
        'aggregation_functions': aggregation_functions,
        'training_set': training,
        'dev_set': dev,
        'test_set': test,
        'normalized_length': normalized_length,
        'labels': labels,
        'num_columns': num_dstat_features,
        'num_features': len(labels)
    }

    # Save the config and complete list of run uuids
    gather_results.save_run_uuids(dataset, runs, data_path=target_data_path,
                                  s3=s3)
    gather_results.save_model_config(dataset, model_config,
                                     data_path=target_data_path, s3=s3)
    print("Stored %d run IDs in the model %s config" % (len(runs), dataset))

    # Resolve the aggregation function names to functions
    resolved_agg_fn = [resolve_aggregation_function(x) for x
                       in aggregation_functions]

    datasets = {}
    # Training must come first so we calculate normalization params
    for data_type in ['training', 'dev', 'test']:
        data, _figure_sizes = prepare_dataset(
            dataset, normalized_length, num_dstat_features, data_type,
            features_regex=features_regex,
            sample_interval=sample_interval, class_label=class_label,
            aggregation_functions=resolved_agg_fn,
            visualize=visualize, data_path=data_path,
            target_data_path=target_data_path, s3=s3)
        datasets[data_type] = data
        examples = data['examples']
        if len(examples) == 0:
            continue

        # Perform dataset-wise normalization
        if data_type == 'training':
            n_examples, normalization_params = normalize_dataset(
                examples, labels)

            # We cache normalization parameters from the training data set
            # to normalize the dev and test set, as well as other input data
            model_config['normalization_params'] = normalization_params
            gather_results.save_model_config(dataset, model_config,
                                             data_path=target_data_path, s3=s3)

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
                                    data_path=target_data_path, s3=s3,
                                    **datasets[data_type])

    # Plot some more figures
    if visualize and not aggregation_functions:
        for n in range(n_examples.shape[0]):
            figure_name = sample_interval + "_%s_" + str(n)
            unrolled_norm_plot = pd.Series(n_examples[n]).plot()
            fig = unrolled_norm_plot.get_figure()
            axes = plt.gca()
            axes.set_ylim([-1, 1])
            fig.savefig(os.sep.join(
                [data_plots_folder] + [figure_name % "normalized"]))
            plt.close(fig)

        df = pd.DataFrame(figure_sizes, columns=['size', 'status'])
        size_plot = df.plot.scatter(x='size', y='status')
        fig = size_plot.get_figure()
        fig.savefig(os.sep.join([data_plots_folder] + ['sizes_by_result.png']))
        plt.close(fig)

    # Store labels to disk
    gather_results.save_dataset(dataset, name='labels',
                                data_path=target_data_path, s3=s3,
                                labels=labels)
    print("Done creating dataset %s" % model_config)


@click.command()
@click.option('--experiment', default='experiment',
              help="Name of the experiment")
@click.option('--estimator', default='tf.estimator.DNNClassifier',
              help='Type of model to be used (not implemented yet).')
@click.option('--hidden-layers', default='10/10/10',
              help='A string that represents the number of layers and units')
@click.option('--steps', default=30,
              help="Hyper param: max number of training steps")
@click.option('--batch-size', default='1',
              help='Hyper param: Number of batches')
@click.option('--epochs', default='1', help='Hyper param: Number of epochs')
@click.option('--optimizer', default='Adagrad',
              type=click.Choice(['Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD',
                                 'ProximalAdagrad']),
              help='Type of optimizer.')
@click.option('--learning-rate', default=0.05, help='Learning rate')
@click.option('--force/--no-force', default=False,
              help='When True, override existing dataset config')
@click.option('--data-path', default=None,
              help="Path to the raw data, local path or s3://<bucket>")
@click.option('--s3-profile', default=None, help='Named configuration')
@click.option('--s3-url',
              default='https://s3.eu-geo.objectstorage.softlayer.net',
              help='Endpoint URL for the s3 storage')
def setup_experiment(experiment, estimator, hidden_layers, steps,
                     batch_size, epochs, optimizer, learning_rate, force,
                     data_path, s3_profile, s3_url):
    """Define experiment parameters and hyper parameters

    Supported optimizers:
    * 'Adagrad': Returns an `AdagradOptimizer`.
    * 'Adam': Returns an `AdamOptimizer`.
    * 'Ftrl': Returns an `FtrlOptimizer`.
    * 'RMSProp': Returns an `RMSPropOptimizer`.
    * 'SGD': Returns a `GradientDescentOptimizer`.
    """
    # s3 support, only for loading the dataset
    s3 = gather_results.get_s3_client(s3_url=s3_url, s3_profile=s3_profile)

    # Prevent overwrite by mistake
    if gather_results.load_experiment(
            experiment, data_path=data_path, s3=s3) and not force:
        print("Experiment %s already configured" % experiment)
        sys.exit(1)
    params = {}
    hyper_params = {
        'steps': steps,
        'batch_size': batch_size,
        'epochs': epochs,
        'hidden_units': [x for x in map(lambda x:int(x),
                                        hidden_layers.split('/'))],
        'optimizer': optimizer,
        'learning_rate': learning_rate
    }
    experiment_data = {
        'estimator': estimator,
        'params': params,
        'hyper_params': hyper_params
    }
    # Store the experiment to disk
    gather_results.save_experiment(experiment_data, experiment,
                                   data_path=data_path, s3=s3)
    print("Experiment %s saved successfully." % experiment)
    print("\testimator: %s" % estimator)
    print("\tparameters: %s" % params)
    print("\thyper parameters: %s" % hyper_params)


@click.command()
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--experiment', default='experiment',
              help="Name of the experiment")
@click.option('--eval-dataset', multiple=True, default=None,
              help='Name of a dataset to be used for alternate evaluation')
@click.option('--gpu/--no-gpu', default=False, help='Force using gpu')
@click.option('--debug/--no-debug', default=False)
@click.option('--data-path', default=None,
              help="Path to the raw data, local path or s3://<bucket>")
@click.option('--s3-profile', default=None, help='Named configuration')
@click.option('--s3-url',
              default='https://s3.eu-geo.objectstorage.softlayer.net',
              help='Endpoint URL for the s3 storage')
def local_trainer(dataset, experiment, eval_dataset, gpu, debug, data_path,
                  s3_profile, s3_url):
    # s3 support. When both using s3, dataset and experiment must stored
    # in the same bucket
    s3 = gather_results.get_s3_client(s3_url=s3_url, s3_profile=s3_profile)

    # Load experiment data
    experiment_data = gather_results.load_experiment(
        experiment, data_path=data_path, s3=s3)
    if not experiment_data:
        print("Experiment %s not found" % experiment)
        sys.exit(1)

    # Load dataset data
    dataset_data = gather_results.load_model_config(
        dataset, data_path=data_path, s3=s3)
    if not dataset_data:
        print("Dataset %s not found" % dataset)
        sys.exit(1)

    # Read hyper_params and params
    estimator = experiment_data['estimator']
    hyper_params = experiment_data['hyper_params']
    params = experiment_data['params']
    steps = int(hyper_params['steps'])
    num_epochs = int(hyper_params['epochs'])
    batch_size = int(hyper_params['batch_size'])
    optimizer = hyper_params['optimizer']
    learning_rate = float(hyper_params['learning_rate'])
    class_label = dataset_data['class_label']

    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)

    # Load the normalized data
    labels = gather_results.load_dataset(dataset, 'labels',
                                         data_path=data_path, s3=s3)['labels']
    training_data = gather_results.load_dataset(dataset, 'training',
                                                data_path=data_path, s3=s3)
    test_data = gather_results.load_dataset(dataset, 'test',
                                            data_path=data_path, s3=s3)
    print("Training data shape: (%d, %d)" % training_data['examples'].shape)

    if class_label == 'node_provider':
        label_vocabulary = set(['rax', 'ovh', 'packethost-us-west-1',
                                'vexxhost', 'limestone-regionone',
                                'inap-mtl01', 'fortnebula-regionone'])
    elif class_label == 'node_provider_all':
        label_vocabulary = set(['rax-iad', 'ovh-bhs1', 'packethost-us-west-1',
                                'rax-dfw', 'vexxhost-ca-ymq-1', 'ovh-gra1',
                                'limestone-regionone', 'inap-mtl01', 'rax-ord',
                                'vexxhost-sjc1', 'fortnebula-regionone'])
    else:
        label_vocabulary = None

    # Get the estimator
    model_dir = gather_results.get_model_folder(dataset, experiment)
    estimator = tf_trainer.get_estimator(
        estimator, hyper_params, params, labels, model_dir,
        optimizer=_OPTIMIZER_CLS_NAMES[optimizer](learning_rate=learning_rate),
        label_vocabulary=label_vocabulary, gpu=gpu)

    def train_and_eval():
        # Train
        tf_trainer.get_training_method(estimator)(
            input_fn=tf_trainer.get_input_fn(shuffle=True,
                                             batch_size=batch_size, num_epochs=num_epochs,
                                             labels=labels, **training_data), steps=steps)
        # Eval on the experiment dataset + any other requested
        eval_sets = [dataset]
        eval_sets.extend(eval_dataset)
        for eval_dataset_name in eval_sets:
            eval_data = gather_results.load_dataset(eval_dataset_name, 'test',
                                                    data_path=data_path, s3=s3)
            eval_size = len(eval_data['example_ids'])

            # Run tf evaluation and store the metrics
            print(
                "Evaluation data shape: (%d, %d)" % eval_data['examples'].shape)
            eval_loss = estimator.evaluate(
                input_fn=tf_trainer.get_input_fn(
                    batch_size=eval_size, num_epochs=1,
                    labels=labels, **eval_data), name=eval_dataset_name)
            # Saving and Logging loss
            print('Training eval data for %s: %r' % (eval_dataset_name,
                                                     eval_loss))
            eval_name = "eval_" + eval_dataset_name
            gather_results.save_data_json(dataset, eval_loss, eval_name,
                                          sub_folder=experiment)

        # Run a prediction on the "dev" set, which we use as prod, and store it
        prod_data = gather_results.load_dataset(dataset, 'dev',
                                                data_path=data_path, s3=s3)
        prod_size = len(prod_data['example_ids'])

        prediction = estimator.predict(
            input_fn=tf_trainer.get_input_fn(
                batch_size=prod_size, num_epochs=1,
                labels=labels, **prod_data))

        # Convert bytes fields to string for serialization
        serializable_pred = []
        for pred in prediction:
            _classes = pred['classes']
            pred['classes'] = [x.decode("utf-8") for x in _classes]
            serializable_pred.append(pred)

        prediction_name = "prediction_" + dataset
        pred_data = zip(prod_data['example_ids'], serializable_pred,
                        prod_data['classes'])
        gather_results.save_data_json(
            dataset, [x for x in pred_data],
            prediction_name, sub_folder=experiment)

    # Now do the training and evalutation
    if gpu:
        with tf.device('/gpu:0'):
            eval_loss = train_and_eval()
    else:
        eval_loss = train_and_eval()


if __name__ == '__main__':
    function_name = sys.argv[1]
    del sys.argv[1]
    globals()[function_name]()
