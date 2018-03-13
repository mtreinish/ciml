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

import tensorflow as tf

BATCH_SIZE = 1000


class SVMTrainer(object):
    def __init__(self, examples, example_ids, labels, classes):
        # Define feature names including the original CSV column name
        self.feature_columns = [tf.contrib.layers.real_valued_column(v + str(k))
                                for k,v in labels.items()]
        self.example_ids = example_ids
        self.examples = examples
        self.classes = classes
        model_data_folder = os.sep.join([
            os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data',
            dataset_name, 'model'])
        os.makedirs(model_data_folder, exist_ok=True)
        self.estimator = estimator(
            feature_columns=self.feature_columns,
            example_id_column='example_id',
            model_dir=model_data_folder,
            feature_engineering_fn=self.feature_engineering_fn)

    def input_fn(self):
        num_examples = len(self.examples)
        # Dict comprehension to build a dict of features
        # I suppose numpy might be able to do this more efficiently
        _features = {self.labels[n]: [x[n] for x in self.examples]
                     for n in range(num_examples)}
        _features['example_id'] = tf.constant(self.example_ids)
        return _features, tf.constant(self.classes)

    def feature_engineering_fn(self, features, labels):
        # Further data normalization may happen here
        return features, labels

    def train(self, steps=30):
        self.estimator.fit(input_fn=self.input_fn, steps=steps)
