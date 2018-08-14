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


class SVMTrainer(object):
    def __init__(self, labels, training_data, eval_data,
                 dataset_name='dataset', force_gpu=False,
                 model_path=None):
        # Define feature names including the original CSV column name
        self.feature_columns = [
            tf.contrib.layers.real_valued_column(x) for x in labels]
        self.training_data = training_data
        self.eval_data = eval_data
        self.force_gpu = force_gpu
        if not model_path:
            model_data_folder = os.sep.join([
                os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data',
                dataset_name, 'model'])
        else:
            model_data_folder = os.sep.join([model_path, 'data', dataset_name,
                                             'model'])
        os.makedirs(model_data_folder, exist_ok=True)
        my_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_secs = 10,  # Save checkpoints every 20 minutes.
            keep_checkpoint_max = 100,       # Retain the 10 most recent checkpoints.
        )
        self.estimator = tf.contrib.learn.SVM(
            feature_columns=self.feature_columns,
            example_id_column='example_id',
            model_dir=model_data_folder,
            config=my_checkpointing_config)

    def input_fn(self, examples, example_ids, classes, return_classes=True):
        num_features = len(self.feature_columns)
        # Dict comprehension to build a dict of features
        # I suppose numpy might be able to do this more efficiently
        _features = {
            self.feature_columns[n].column_name:
                tf.constant(examples)
            for n in range(num_features)}
        _features['example_id'] = tf.constant(example_ids)
        print("Done preparing input data")
        if return_classes:
            return _features, tf.constant(classes)
        else:
            return _features

    def training_input_fn(self):
        return self.input_fn(**self.training_data)

    def evaluate_input_fn(self):
        return self.input_fn(**self.eval_data)

    def train(self, steps=30):
        if self.force_gpu:
            with tf.device('/device:GPU:0'):
                self.estimator.fit(input_fn=self.training_input_fn,
                                   steps=steps)
                train_loss = self.estimator.evaluate(
                    input_fn=self.evaluate_input_fn, steps=1)
        else:
            self.estimator.fit(input_fn=self.training_input_fn, steps=steps)
            train_loss = self.estimator.evaluate(
                input_fn=self.evaluate_input_fn, steps=1)
        print('Training loss %r' % train_loss)

    def predict_fn(self):
        return self.input_fn(return_classes=False, **self.training_data)

    def predict(self):
        prediction = list(self.estimator.predict(input_fn=self.predict_fn))
        return prediction
