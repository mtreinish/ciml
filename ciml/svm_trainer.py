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
    def __init__(self, examples, example_ids, labels, classes,
                 dataset_name='dataset', force_gpu=False):
        # Define feature names including the original CSV column name
        self.feature_columns = [
            tf.contrib.layers.real_valued_column(x) for x in labels]
        self.example_ids = np.array(example_ids)
        self.examples = examples
        self.classes = classes
        self.force_gpu = force_gpu
        model_data_folder = os.sep.join([
            os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data',
            dataset_name, 'model'])
        os.makedirs(model_data_folder, exist_ok=True)
        self.estimator = tf.contrib.learn.SVM(
            feature_columns=self.feature_columns,
            example_id_column='example_id',
            model_dir=model_data_folder,
            feature_engineering_fn=self.feature_engineering_fn)

        # Separate traing set and evaluation set by building randomised lists
        # of indexes that can be used for examples, example_ids and classes
        self.all_indexes = range(len(self.example_ids))
        self.training_idx = pd.Series(self.all_indexes).sample(
            len(self.example_ids) // 2).values
        self.evaluate_idx = list(
            set(self.all_indexes) - set(self.training_idx))

    def input_fn(self, idx_filter, return_classes=True):
        num_features = len(self.feature_columns)
        # Dict comprehension to build a dict of features
        # I suppose numpy might be able to do this more efficiently
        _features = {
            self.feature_columns[n].column_name:
                tf.constant(self.examples[idx_filter, n])
            for n in range(num_features)}
        _features['example_id'] = tf.constant(self.example_ids[idx_filter])
        print("Done preparing input data")
        if return_classes:
            return _features, tf.constant(self.classes[idx_filter])
        else:
            return _features

    def training_input_fn(self):
        return self.input_fn(self.training_idx)

    def evaluate_input_fn(self):
        return self.input_fn(self.evaluate_idx)

    def feature_engineering_fn(self, features, labels):
        # Further data normalization may happen here
        print("Built engineered data")
        return features, labels

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
        return self.input_fn(range(len(self.example_ids)),
                             return_classes=False)

    def predict(self):
        prediction = list(self.estimator.predict(input_fn=self.predict_fn))
        return prediction
