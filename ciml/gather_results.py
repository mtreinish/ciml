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

import datetime
import gzip
import io
import json
import os

import pandas
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from subunit2sql.db import api

now = datetime.datetime.utcnow()


def _parse_dstat_date(date_str):
    split_date_time = date_str.split(' ')
    monthday = split_date_time[0].split('-')
    time_pieces = split_date_time[1].split(':')
    return datetime.datetime(now.year, int(monthday[1]), int(monthday[0]),
                             int(time_pieces[0]),
                             int(time_pieces[1]), int(time_pieces[2]))


def _parse_dstat_file(input_io, sample_interval=None):
    """Parse a single dstat file into a DatetimeIndex.

    Parse a dstat file into a DatetimeIndex.
    Optionally resample to the specified sample_interval.

    A dstat file is a "rolled" example with size s x d, where:
    - s is the number of samples (over time) after resampling
    - d is the number of dstat columns available
    """
    out = pandas.read_csv(input_io, skiprows=6).set_index('time')
    out.index = [_parse_dstat_date(x) for x in out.index]
    out.index = pandas.DatetimeIndex(out.index)
    if sample_interval:
        out = out.resample(sample_interval).mean()
    return out


def _get_dstat_file(artifact_link, run_uuid=None, sample_interval=None):
    """Obtains and parses a dstat file to a pandas.DatetimeIndex

    Finds a dstat file in the local cache or downloads it from the
    artifacts link, then parses it and resamples it into a pandas.DatetimeIndex.
    """
    paths = ['controller/logs/dstat-csv_log.txt.gz',
             'controller/logs/dstat-csv_log.txt',
             'logs/dstat-csv_log.txt',
             'logs/dstat-csv_log.txt.gz']
    raw_data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                       'data', '.raw']
    os.makedirs(os.sep.join(raw_data_folder), exist_ok=True)
    raw_data_file = os.sep.join(raw_data_folder + [run_uuid + '.csv.gz'])
    if os.path.isfile(raw_data_file):
        try:
            with gzip.open(raw_data_file, mode='r') as f:
                return _parse_dstat_file(f, sample_interval)
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load this run.
            print('Run %s found in the local dataset, however: %s',
                  (run_uuid, ioe))
            return None

    # If no local cache was found we try to fetch the dstats file via HTTP
    # and we store it in cache.
    for path in paths:
        url = artifact_link + '/' + path
        resp = requests.get(url)
        if resp.status_code == 404:
            continue
        # Cache the file locally
        with gzip.open(raw_data_file, mode='wb') as local_cache:
            local_cache.write(resp.text.encode())
        # And return the parse dstat
        f = io.StringIO(resp.text)
        return _parse_dstat_file(f, sample_interval)
    else:
        return None


def _get_result_for_run(run, session, use_cache=True):
    # First try to get the data from disk
    metadata_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                       'data', '.metadata']
    os.makedirs(os.sep.join(metadata_folder), exist_ok=True)
    result_file = os.sep.join(metadata_folder + [run.uuid + '.json.gz'])
    if use_cache:
        if os.path.isfile(result_file):
            try:
                with gzip.open(result_file, mode='r') as f:
                    return json.loads(f.read())
            except IOError as ioe:
               # Something went wrong opening the file, so we won't load this run.
                print('Run %s found in the local dataset, however: %s',
                      (run.uuid, ioe))
                return None
    # If no local cache, get data from the DB
    result = {}
    test_runs = api.get_test_runs_by_run_id(run.uuid, session=session)
    tests = []
    for test_run in test_runs:
        test = {'status': test_run.status}
        start_time = test_run.start_time
        start_time = start_time.replace(
            microsecond=test_run.start_time_microsecond)
        stop_time = test_run.stop_time
        stop_time = stop_time.replace(
            microsecond=test_run.stop_time_microsecond)
        test['start_time'] = start_time
        test['stop_time'] = stop_time
        tests.append(test)
    if run.fails > 0 or run.passes == 0:
        result['status'] = 1  # Failed
    else:
        result['status'] = 0  # Passed
    result['artifact'] = run.artifacts
    # Get extra run metadata
    metadata = api.get_run_metadata(run.uuid, session)
    for md in metadata:
        result[md['key']] = md['value']
    # Cache the json file, without tests
    with gzip.open(result_file, mode='wb') as local_cache:
        local_cache.write(json.dumps(result).encode())
    result['tests'] = tests
    return result


def _get_data_for_run(run, sample_interval, session, use_cache=True):
    # First ensure we can get dstat data
    dstat = _get_dstat_file(run.artifacts, run.uuid, sample_interval)
    if dstat is None:
        return None
    result = _get_result_for_run(run, session, use_cache)
    result['dstat'] = dstat
    return result


def get_subunit_results(build_uuid, dataset_name, sample_interval, db_uri,
                        build_name='tempest-full', use_cache=True):
    engine = create_engine(db_uri)
    Session = sessionmaker(bind=engine)
    session = Session()
    runs = api.get_runs_by_key_value('build_uuid', build_uuid, session=session)
    results = []
    for run in runs:
        # Check if we are interested in this build at all
        meta = api.get_run_metadata(run.uuid, session=session)
        build_names = [x.value for x in meta if x.key == 'build_name']
        if len(build_names) >= 1:
            # Skip build_names that aren't selected
            if not build_name == build_names[0]:
                continue
            db_build_name = build_names[0]
        else:
            continue
        # NOTE(mtreinish): Only be concerned with single node to start
        if 'multinode' in db_build_name:
            continue
        result = _get_data_for_run(run, sample_interval, session, use_cache)
        if result:
            results.append(result)
    session.close()
    return results


def get_subunit_results_for_run(run, sample_interval, db_uri=None):
    if db_uri:
        # When running from a local set the db_uri is not going to be set
        engine = create_engine(db_uri)
        Session = sessionmaker(bind=engine)
        session = Session()
    else:
        session = None
    return [_get_data_for_run(run, sample_interval, session)]


def get_runs_by_name(db_uri, build_name):
    engine = create_engine(db_uri)
    Session = sessionmaker(bind=engine)
    session = Session()
    runs = api.get_runs_by_key_value('build_name', build_name, session=session)
    return runs


def save_model_config(dataset, model_config):
    data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                   'data', dataset]
    os.makedirs(os.sep.join(data_folder), exist_ok=True)
    model_config_file = os.sep.join(data_folder + [dataset + '.json.gz'])
    existing_config = load_model_config(dataset)
    # TODO(andreaf) For now we just override things. This would actually be a
    # good place to fail or at least warn users that the model is being
    # re-trained with conflicting parameters.
    if existing_config:
        existing_config.update(model_config)
        model_config = existing_config

    with gzip.open(model_config_file, mode='wb') as local_cache:
        local_cache.write(json.dumps(model_config).encode())


def load_model_config(dataset):
    data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                   'data', dataset]
    model_config_file = os.sep.join(data_folder + [dataset + '.json.gz'])
    if os.path.isfile(model_config_file):
        try:
            with gzip.open(model_config_file, mode='r') as f:
                return json.loads(f.read())
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load this run.
            print('Dataset config found in the local dataset, however: %s', ioe)
            return None


def load_run_uuids(dataset):
    """Return a list of run objects for a specific dataset_name

    Read the list of run uuids from file and return a list of run objects
    compatible with the run returned by the DB api. The only valid content
    in the run objects is the UUID.
    """
    # TODO(andreaf) We can cache run data in the .metadata folder as well and
    # build full run objects here. Alternatively we could build dictionaries
    # and change gather_results functions to work on the dict as opposed to the
    # object.
    class _run(object):
        def __init__(self, uuid):
            self.uuid = uuid
            self.artifacts = None

    dataset_runs = os.sep.join([os.path.dirname(os.path.realpath(__file__)),
                                os.pardir, 'data', dataset, 'runs.json.gz'])
    if os.path.isfile(dataset_runs):
        try:
            with gzip.open(dataset_runs, mode='r') as f:
                return [_run(run_uuid) for run_uuid in json.loads(f.read())]
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load this run.
            print('Run %s found in the local dataset, however: %s',
                  (run.uuid, ioe))
            return None


def save_run_uuids(dataset, runs):
    dataset_runs = os.sep.join([os.path.dirname(os.path.realpath(__file__)),
                                os.pardir, 'data', dataset, 'runs.json.gz'])
    run_uuids = [run.uuid for run in runs]
    with gzip.open(dataset_runs, mode='wb') as local_cache:
        local_cache.write(json.dumps(run_uuids).encode())
