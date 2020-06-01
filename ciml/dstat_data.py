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

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

BATCH_SIZE = 1000


class DstatTrainer(object):
    def __init__(self, dataset_name, estimator=tf.estimator.DNNClassifier):
        self.metric_columns = [
            # total cpu usage
            tf.feature_column.numeric_column(key='usr'),
            tf.feature_column.numeric_column(key='sys'),
            tf.feature_column.numeric_column(key='idl'),
            tf.feature_column.numeric_column(key='wai'),
            tf.feature_column.numeric_column(key='sig'),
            # memory usage
            tf.feature_column.numeric_column(key='used'),
            tf.feature_column.numeric_column(key='buff'),
            tf.feature_column.numeric_column(key='cach'),
            tf.feature_column.numeric_column(key='free'),
            # net total
            tf.feature_column.numeric_column(key='recv'),
            tf.feature_column.numeric_column(key='send'),
            # disk total
            tf.feature_column.numeric_column(key='read'),
            tf.feature_column.numeric_column(key='writ'),
            # io total
            tf.feature_column.numeric_column(key='read.1'),
            tf.feature_column.numeric_column(key='writ.1'),
            # system
            tf.feature_column.numeric_column(key='int'),
            tf.feature_column.numeric_column(key='csw'),
            # load average
            tf.feature_column.numeric_column(key='1m'),
            tf.feature_column.numeric_column(key='5m'),
            tf.feature_column.numeric_column(key='15m'),
            # procs
            tf.feature_column.numeric_column(key='run'),
            # paging
            tf.feature_column.numeric_column(key='blk'),
            tf.feature_column.numeric_column(key='new'),
            # tcp sockets
            tf.feature_column.numeric_column(key='in'),
            tf.feature_column.numeric_column(key='out'),
            tf.feature_column.numeric_column(key='lis'),
            tf.feature_column.numeric_column(key='act'),
            tf.feature_column.numeric_column(key='syn'),
            tf.feature_column.numeric_column(key='tim'),
            tf.feature_column.numeric_column(key='clo'),
        ]
        model_data_folder = os.sep.join([
            os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data',
            dataset_name, 'model'])
        os.makedirs(model_data_folder, exist_ok=True)
        self.estimator = estimator(feature_columns=self.metric_columns,
                                   hidden_units=[31, 31],
                                   n_classes=2,
                                   model_dir=model_data_folder)

    def train(self, dstat_data, status):

        def _train_input_function(features, labels, batchsize):
            # NOTE(andreaf) Not sure where to fit the test result
            # dataset = tf.data.Dataset.from_tensor_slices(
            #     dict(features), status)
            dataset = tf.data.Dataset.from_tensor_slices(dict(features))
            dataset = dataset.batch(batchsize)
            return dataset

        self.estimator.train(input_fn=lambda: _train_input_function(
            dstat_data, status, BATCH_SIZE))
