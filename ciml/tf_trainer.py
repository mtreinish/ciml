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
    return [tf.contrib.layers.real_valued_column(x) for x in labels]


def input_fn(examples, example_ids, classes, labels):
    feature_columns = get_feature_columns(labels)
    num_features = len(labels)
    # Dict comprehension to build a dict of features
    # I suppose numpy might be able to do this more efficiently
    _features = {
        labels[n]: tf.constant(examples[:,n]) for n in range(num_features)}
    # _features['example_id'] = tf.constant(example_ids)
    return _features, tf.constant(classes)


def get_checkpoint_config():
    return tf.estimator.RunConfig(
        save_checkpoints_secs = 10,  # Save checkpoints every 10s.
        keep_checkpoint_max = 100,   # Retain the 100 most recent checkpoints.
    )

def get_estimator(estimator_name, hyper_params, params, labels, model_dir):

    # SVM Model
    if estimator_name == 'tf.contrib.learn.SVM':
        return tf.contrib.learn.SVM(
            feature_columns=get_feature_columns(labels),
            example_id_column='example_id',
            model_dir=model_dir,
            config=get_checkpoint_config())

    # DNN Model
    if estimator_name == 'tf.estimator.DNNClassifier':
        return tf.estimator.DNNClassifier(
            feature_columns=get_feature_columns(labels),
            model_dir=model_dir,
            config=get_checkpoint_config(),
            hidden_units=hyper_params['hidden_units'],
            n_classes=2)

def get_training_method(estimator):
    if type(estimator).__name__ == 'SVM':
        return getattr(estimator, 'fit')
    else:
        return getattr(estimator, 'train')
