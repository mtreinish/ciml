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

import argparse
from contextlib import contextmanager
import os
import threading

import flask
from flask import abort
from flask.json import jsonify
from flask import make_response
import numpy as np
from six.moves import configparser as ConfigParser
from six.moves.urllib import parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


from ciml import gather_results
from ciml import nn_trainer
from ciml import svm_trainer
from ciml.trainer import default_db_uri
from ciml.trainer import fixed_lenght_example
from ciml.trainer import get_class
from ciml.trainer import normalize_dataset
from ciml.trainer import unroll_example


app = flask.Flask('model-train')
app.config['PROPAGATE_EXCEPTIONS'] = True
config = None
db_uri = None
engine = None
Session = None
model_dir = None
estimator = None


def get_app():
    return app


@app.before_first_request
def _setup():
    setup()


def parse_command_line_args():
    description = 'Starts the API service for ciml-train-api'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('config_file', type=str, nargs='?',
                        default='/etc/ciml-train.conf',
                        help='the path for the config file to be read.')
    return parser.parse_args()


def _config_get(config_func, section, option, default_val=False):
    retval = default_val
    if default_val is not False:
        try:
            retval = config_func(section, option)
        except ConfigParser.NoOptionError:
            pass
    else:
        retval = config_func(section, option)
    return retval


def setup():
    global config
    if not config:
        args = parse_command_line_args()
        config = ConfigParser.ConfigParser()
        config.read(args.config_file)
    global model_dir
    model_dir = _config_get(config.get, 'default', 'model_dir',
                            '/shared/model')
    global estimator
    estimator = _config_get(config.get, 'default', 'estimator', 'svm')
    if estimator not in ['svm', 'nn']:
        raise TypeError("Configured model estimator %s is not a valid choice."
                        " It must either be 'svm' or 'nn'")
    # DB session setup
    global engine
    db_uri = _config_get(config.get, 'default', 'db_uri', default_db_uri)
    pool_size = _config_get(config.getint, 'default', 'pool_size', 20)
    pool_recycle = _config_get(config.getint, 'default', 'pool_recycle', 3600)
    engine = create_engine(db_uri,
                           pool_size=pool_size,
                           pool_recycle=pool_recycle)
    global Session
    Session = sessionmaker(bind=engine)


def get_session():
    global Session
    if not Session:
        setup()
    return Session()


@contextmanager
def session_scope():
    try:
        session = get_session()
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@app.route('/', methods=['GET'])
def list_routes():
    output = []
    for rule in app.url_map.iter_rules():
        options = {}
        for arg in rule.arguments:
            options[arg] = "[{0}]".format(arg)
        url = flask.url_for(rule.endpoint, **options)
        out_dict = {
            'name': rule.endpoint,
            'methods': sorted(rule.methods),
            'url': parse.unquote(url),
        }
        output.append(out_dict)
    return jsonify({'routes': output})


@app.route('/train/name/<string:build_name>', methods=['POST'])
def train_model(build_name):

    global estimator
    dataset = estimator
    global model_dir
    with session_scope() as session:
        if not os.path.isfile(os.sep.join([model_dir, 'data', dataset,
                                           'runs.json.gz'])):
            runs = gather_results.get_runs_by_name(None,
                                                   build_name=build_name,
                                                   session=session)
            model_config = {'build_name': build_name}
            gather_results.save_model_config(dataset, model_config,
                                             data_path=model_dir)
            gather_results.save_run_uuids(dataset, runs,
                                          data_path=model_dir)
        else:
            runs = gather_results.load_run_uuids(dataset,
                                                 data_path=model_dir)
        normalized_length = 5500
        if estimator == 'svm':
            skips = []
            classes = []
            labels = []
            examples = []
            class_label = 'status'
            features_regex = None
            sample_interval = None
            idx = 0
            # Model configuration. We need to cache sample_interval,
            # features-regex and the normalization parameters for each
            # feature so we can re-use them during prediction.
            model_config = {
                'sample_interval': sample_interval,
                'features_regex': features_regex,
                'normalized_length': normalized_length
            }
            for run in runs:
                results = gather_results.get_subunit_results_for_run(
                    run, '1s', session=None, data_path=model_dir,
                    use_cache=True)
                print('Acquired run %s' % run.uuid)
                # For one run_uuid we must only get on example (result)
                result = results[0]
                if not result:
                    skips.append(run.uuid)
                    continue
                # Setup the numpy matrix and sizes
                if len(examples) == 0:
                    # Adjust normalized_length to the actual re-sample one
                    examples = np.ndarray(
                        shape=(
                            len(runs),
                            (len(result['dstat'].columns)
                             * normalized_length)))
                    model_config['num_columns'] = len(
                        result['dstat'].columns)
                    model_config['num_features'] = (len(
                        result['dstat'].columns) * normalized_length)
                    # Normalize data
                    example = fixed_lenght_example(result,
                                                   normalized_length)
                    # Normalize status
                    status = get_class(result, class_label)
                    vector, new_labels = unroll_example(
                        example, normalized_length, labels)
                    # Only calculate labels for the first example
                    if len(labels) == 0:
                        labels = new_labels
                        model_config['labels'] = labels
                    # Examples is an np ndarrays
                    examples[idx] = vector.values
                    classes.append(status)
            if len(skips) > 0:
                    print('Unable to train model because of missing '
                          'runs %s' % skips)
                    safe_runs = [
                        run for run in runs if run.uuid not in skips]
                    gather_results.save_run_uuids(dataset, safe_runs,
                                                  data_path=model_dir)
                    message = ('The model has been updated to exclude '
                               'those runs. Please re-run the training'
                               ' step.')
                    abort(make_response(message, 400))

            def run_training():
                # Perform dataset-wise normalization
                # NOTE(andreaf) When we train the model we ignore any saved
                # normalization
                # parameter, since the sample interval and features may be
                # different.
                n_examples, normalization_params = normalize_dataset(
                    examples, labels)
                # We do cache the result to normalize the prediction set.
                model_config['normalization_params'] = normalization_params
                gather_results.save_model_config(dataset, model_config,
                                                 data_path=model_dir)
                # Now do the training
                example_ids = [run.uuid for run in runs]
                outclasses = np.array(classes)
                svm_trainer.SVMTrainer(n_examples, example_ids, labels,
                                       outclasses, dataset_name=dataset,
                                       model_path=model_dir)
            thread = threading.Thread(target=run_training)
            thread.start()
            return "training started", 202
        else:
            def run_nn_training():
                for run in runs:
                    result = gather_results.get_subunit_results_for_run(
                        run, '1s', session=session, use_cache=False,
                        data_path=model_dir)[0]
                    try:
                        features, labels = nn_trainer.normalize_data(result)
                    except TypeError:
                        print('Unable to normalize data in run %s, '
                              'skipping' % run.uuid)
                        continue
                    nn_trainer.train_model(features, labels,
                                           dataset_name=dataset,
                                           model_path=model_dir)
            thread = threading.Thread(target=run_nn_training)
            thread.start()
            return "training started", 202


def main():
    global config
    args = parse_command_line_args()
    config = ConfigParser.ConfigParser()
    config.read(args.config_file)
    try:
        host = config.get('default', 'host')
    except ConfigParser.NoOptionError:
        host = '127.0.0.1'
    try:
        port = config.getint('default', 'port')
    except ConfigParser.NoOptionError:
        port = 5000
    app.run(debug=True, host=host, port=port)


if __name__ == '__main__':
    main()
