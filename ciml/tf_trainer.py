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

import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def get_feature_columns(labels):
    return [tf.feature_column.numeric_column(x) for x in labels]


def get_input_fn(examples, example_ids, classes, labels,
                 shuffle=False, batch_size=None, num_epochs=None):
    """Generate an input_fn to feed the estimator

    Input: examples, classes, labels, batch size, num epochs
    Output: a function that emits data for the estimator
    """
    _numpy_input_fn = tf.estimator.inputs.numpy_input_fn
    # Dict comprehension to build a dict of features
    # I suppose numpy might be able to do this more efficiently
    _features = {
        labels[n]: examples[:,n] for n in range(len(labels))}
    params = {}
    if batch_size: params['batch_size'] = batch_size
    if num_epochs: params['num_epochs'] = num_epochs
    return _numpy_input_fn(_features, classes, shuffle=shuffle, **params)


def get_estimator_config(gpu):
    config_params = {
        'save_checkpoints_secs': 300,  # Save checkpoints every 5min.
        'keep_checkpoint_max': 10,   # Retain the 10 most recent checkpoints.
    }
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    if gpu:
        session_config.gpu_options.allow_growth = True
    config_params['session_config'] = session_config

    return tf.estimator.RunConfig(**config_params)

def get_estimator(estimator_name, hyper_params, params, labels, model_dir,
                  optimizer=None, label_vocabulary=None, gpu=False):

    estimator_params = {}
    if optimizer:
        estimator_params['optimizer'] = optimizer

    # If no vocabulary is passed, we assume 2 classes with 0 and 1 values
    estimator_params['n_classes'] = 2
    if label_vocabulary:
        estimator_params['n_classes'] = len(label_vocabulary)
        estimator_params['label_vocabulary'] = list(label_vocabulary)
    # SVM Model
    if estimator_name == 'tf.contrib.learn.SVM':
        return tf.contrib.learn.SVM(
            feature_columns=get_feature_columns(labels),
            example_id_column='example_id',
            model_dir=model_dir,
            config=get_estimator_config(gpu=gpu),
            **estimator_params)

    # DNN Model
    if estimator_name == 'tf.estimator.DNNClassifier':
        estimator = tf.estimator.DNNClassifier(
            feature_columns=get_feature_columns(labels),
            model_dir=model_dir,
            config=get_estimator_config(gpu=gpu),
            hidden_units=hyper_params['hidden_units'],
            **estimator_params)
        return estimator


def get_training_method(estimator):
    if type(estimator).__name__ == 'SVM':
        return getattr(estimator, 'fit')
    else:
        return getattr(estimator, 'train')
