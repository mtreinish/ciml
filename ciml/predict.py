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
import re
import sys

import click
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from ciml import gather_results
from ciml import listener
from ciml import svm_trainer
from ciml import trainer


default_mqtt_hostname = ('firehose.openstack.org')
default_db_uri = ('mysql+pymysql://query:query@logstash.openstack.org/'
                  'subunit2sql')


@click.command()
@click.option('--db-uri', default=default_db_uri, help="DB URI")
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--sample-interval', default='1s',
              help='dstat (down)sampling interval')
@click.option('--build-name', default="tempest-full", help="Build name.")
@click.option('--debug/--no-debug', default=False)
@click.argument('build_uuid')
def db_predict(db_uri, dataset, sample_interval, build_name, debug,
               build_uuid):
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    results = gather_results.get_subunit_results(
        build_uuid, dataset, sample_interval, db_uri, build_name)
    if results:
        print('Obtained dstat file for %s' % build_uuid)
    else:
        print('Build uuid: %s is not of proper build_uuid, skipping'
              % build_uuid)
    for res in results:
        vector, status, labels = trainer.normalize_example(res)
        model = svm_trainer.SVMTrainer(
            vector, [build_uuid] * len(results), labels, [status],
            dataset_name=dataset)
        model.predict()


@click.command()
@click.option('--db-uri', default=default_db_uri, help="DB URI")
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--slice', default=":", help="Slice of the dataset")
@click.option('--gpu', default=False, help='Force using gpu')
@click.option('--debug/--no-debug', default=False)
def db_batch_predict(db_uri, dataset, slice, gpu, debug):
    """Run predict on all DB items on included in the dataset yet

    Takes a dataset and a build name. It builds the list of runs in the DB
    that fit the specified build name, and that are not yet used for training
    in the specified dataset. It runs prediction on all of them.
    """
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    # Get the configuration for the model
    model_config = gather_results.load_model_config(dataset)
    # Get the list of runs from the dataset
    run_uuids = gather_results.load_run_uuids(dataset)
    # Get the list of runs from the DB
    runs = gather_results.get_runs_by_name(
        db_uri=db_uri, build_name=model_config['build_name'])
    # Run a predict loop, include all runs not in the train dataset
    predict_runs = [r for r in runs if r.uuid not in run_uuids]
    predict_runs = predict_runs[]
    if len(predict_runs) == 0:
        print("Empty prediction set, nothing to do.")
        sys.exit(0)
    # Initialize the array
    examples = np.ndarray(
        shape=(len(predict_runs), model_config['num_features']))
    idx = 0
    classes = []
    labels = []
    print("All runs: %d, dataset size: %d, predict size: %d" % (
        len(runs), len(run_uuids), len(predict_runs)))
    for run in predict_runs:
        # This will also store new runs in cache. In future we may want to
        # train on those as well, but nor now let's try to predict only
        results = gather_results.get_subunit_results_for_run(
            run, model_config['sample_interval'], db_uri=db_uri)
        for result in results:
            # Skip runs with no data
            if result is None:
                continue
            if model_config['features_regex']:
                df = result['dstat']
                col_regex = re.compile(model_config['features_regex'])
                result['dstat'] = df[list(filter(
                    col_regex.search, df.columns))]
            # Normalize examples
            vector, status, labels = trainer.normalize_example(
                result, model_config['normalized_length'],
                model_config['labels'])
            examples[idx] = vector.values
            classes.append(status)
            idx += 1
    # Normalize dataset
    n_examples, _ = trainer.normalize_dataset(
        examples, labels, params=model_config['normalization_params'])
    # Prepare other arrays
    classes = np.array(classes)
    run_uuids = [r.uuid for r in predict_runs]
    # Configure TF
    config = tf.ConfigProto(log_device_placement=True,)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # Now do the prediction
    model = svm_trainer.SVMTrainer(n_examples, run_uuids, labels,
                                   classes, dataset_name=dataset,
                                   force_gpu=gpu)
    predictions = model.predict()
    errors = []
    for prediction, actual in zip(predictions, classes):
        if prediction['classes'] != actual:
            errors.append((prediction, actual))
    print("Prediction of %d inputs completed." % len(classes))
    print("Input set composition: %d PASS, %s FAIL" % (
        len([x for x in classes if x == 0]),
        len([x for x in classes if x == 1])))
    if len(errors) > 0:
        print("There were some prediction errors: %s" % errors)
    else:
        print("All predicted correctly.")


@click.command()
@click.option('--db-uri', default=default_db_uri, help="DB URI")
@click.option('--mqtt-hostname', default=default_mqtt_hostname,
              help='MQTT hostname')
@click.option('--topic', default='gearman-subunit/#',
              help='MQTT topic to subscribe to')
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--sample-interval', default='1s',
              help='dstat (down)sampling interval')
@click.option('--build-name', default="tempest-full", help="Build name.")
@click.option('--debug/--no-debug', default=False)
@click.option('--model-dir', default=None, help='Base path to model dir')
def mqtt_predict(db_uri, mqtt_hostname, topic, dataset, sample_interval,
                 build_name, debug, model_dir):
    event_queue = queue.Queue()
    if debug:
        print('Starting MQTT listener')
    listen_thread = listener.MQTTSubscribe(event_queue, mqtt_hostname, topic)
    listen_thread.start()
    if debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
        print('Entering main loop')
    while True:
        event = event_queue.get()
        if debug:
            print('Received event with build uuid %s' % event['build_uuid'])
        results = gather_results.get_subunit_results(
            event['build_uuid'], dataset, sample_interval, db_uri, build_name,
            data_path=model_dir, use_cache=False)
        if results:
            print('Obtained dstat file for %s' % event['build_uuid'])
        else:
            print('Build uuid: %s is not of proper build_name, skipping'
                  % event['build_uuid'])
        for res in results:
            vector, status, labels = trainer.normalize_example(res)
            model = svm_trainer.SVMTrainer(
                vector, [event['build_uuid']] * len(results), labels, [status],
                dataset_name=dataset, model_path=model_dir)
            model.predict()
