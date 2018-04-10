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


"""Use a DNN to attempt to classify results

The basic expected workflow here is basically::

  result = gather_results._get_result_for_run(...)
  features, labels = nn_trainer.normalize_data(result)
  nn_trainer.train_model(features, labels)

Which will train the model a similar workflow is used for prediction.
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from ciml import trainer

tf.logging.set_verbosity(tf.logging.INFO)


def build_success_time_series(result):
    if not result or result.get('dstat', None) is None or not result.get(
        'tests', None):
        return None
    dstat = result['dstat']
    tests = result['tests']
    fails = [x['stop_time'] for x in tests if x['status'] == 'fail']
    if fails:
        output = []
        first_fail = min(fails)
        for i in dstat.index:
            if i >= first_fail:
                output.append(1)
            else:
                output.append(0)
        return np.array(output)
    else:
        return np.array([0] * len(dstat))


def normalize_status(status, normalized_length=5500):
    if len(status) == normalized_length:
        return status
    elif len(status) > normalized_length:
        return status[:normalized_length]
    else:
        pad_length = normalized_length - len(status)
        padd = np.array([status[-1]] * pad_length)
        return np.concatenate((status, padd))


def pad_start(features, status, pad_length=20):
    status = np.concatenate((np.zeros(pad_length), status))
    feature_pad = {x: np.zeros(pad_length) for x in features.keys()}
    features = pd.concat((pd.DataFrame(feature_pad), features))
    return features, status


def normalize_data(result, normalized_length=5500, pad_start=False):
    status = build_success_time_series(result)
    if status is None:
        raise TypeError
    if result['status'] not in status:
        raise Exception
    features = trainer.fixed_lenght_example(result, normalized_length)
    status = normalize_status(status, normalized_length)
    if pad_start:
        features, status = pad_start(features, status)
    return features, status


def data_input_fn(features, labels, batch_size):
    """An input function for training sliding window time series data"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(features), labels)).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def time_series_model(features, labels, mode, params):
    """DNN custom model.

    :param features: batch_features from input_fn
    :param labels: batch_labels from input_fn
    :param mode: An instance of tf.estimator.ModeKeys
    :param params: Additional configuration dict
    """
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.sigmoid)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_model(data, results, train_steps=1000, dataset_name='dataset'):
    my_feature_columns = [
        tf.contrib.layers.real_valued_column(x) for x in data.keys()]
    model_data_dir = os.sep.join([
        os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data',
        dataset_name, 'model'])
    data = data.reset_index()
    data = data.drop(columns=['index'])

    my_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=data,
        y=pd.Series(results, index=data.index),
        shuffle=False,
        batch_size=100,
        num_threads=1)
    classifier = tf.estimator.DNNClassifier(
        model_dir=model_data_dir,
        feature_columns=my_feature_columns,
        hidden_units=[250, 250, 250, 250, 250],
        n_classes=2)
    classifier.train(
        input_fn=my_input_fn,
        steps=train_steps)


def evaluate_model(data, dataset_name='dataset'):
    my_feature_columns = [
        tf.contrib.layers.real_valued_column(x) for x in data.keys()]
    model_data_dir = os.sep.join([
        os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data',
        dataset_name, 'model'])
    data = data.reset_index()
    data = data.drop(columns=['index'])

    my_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=data,
        shuffle=False,
        batch_size=100,
        num_threads=1)
    classifier = tf.estimator.DNNClassifier(
        model_dir=model_data_dir,
        feature_columns=my_feature_columns,
        hidden_units=[250, 250, 250, 250, 250],
        n_classes=2)
    train_loss = classifier.evaluate(input_fn=my_input_fn, steps=1)
    print('Training loss %r' % train_loss)


def predict_model(data, dataset_name='dataset'):
    model_data_dir = os.sep.join([
        os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data',
        dataset_name, 'model'])
    my_feature_columns = [
        tf.contrib.layers.real_valued_column(x) for x in data.keys()]
    classifier = tf.estimator.DNNClassifier(
        model_dir=model_data_dir,
        feature_columns=my_feature_columns,
        hidden_units=[250, 250, 250, 250, 250],
        n_classes=2)
    my_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=data,
        shuffle=False,
        batch_size=100,
        num_threads=1)
    predictions = list(classifier.precict(input_fn=my_input_fn))
    print("Predicted %s" % predictions)
