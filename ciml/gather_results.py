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
    out = pandas.read_csv(input_io, skiprows=6).set_index('time')
    out.index = [_parse_dstat_date(x) for x in out.index]
    out.index = pandas.DatetimeIndex(out.index)
    if sample_interval:
        out = out.resample(sample_interval).mean()
    return out


def _get_dstat_file(artifact_link, dataset_name, run_uuid=None,
                    sample_interval=None):
    paths = ['controller/logs/dstat-csv_log.txt.gz',
             'controller/logs/dstat-csv_log.txt',
             'logs/dstat-csv_log.txt',
             'logs/dstat-csv_log.txt.gz']
    # TODO(andreaf) This needs to be fixed because when we come different
    # routeswe will need to lookup the file from file system or not, and
    # behave differently:
    # - MQTT: new file, file should not be there
    # - DB: file may be there or not
    # - Stable dataset: file must be there
    raw_data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                       'data', dataset_name, 'raw']
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


def _get_result_for_run(run, dataset_name, session):
    # First try to get the data from disk
    raw_data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                       'data', dataset_name, 'raw']
    os.makedirs(os.sep.join(raw_data_folder), exist_ok=True)
    result_file = os.sep.join(raw_data_folder + [run.uuid + '.json.gz'])
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
        result['status'] = 1 # Failed
    else:
        result['status'] = 0 # Passed
    result['artifact'] = run.artifacts
    # Cache the json file, without tests
    with gzip.open(result_file, mode='wb') as local_cache:
        local_cache.write(json.dumps(result).encode())
    result['tests'] = tests
    return result


def _get_data_for_run(run, dataset_name, sample_interval, session):
    # First ensure we can get dstat data
    dstat = _get_dstat_file(run.artifacts, dataset_name, run.uuid,
                            sample_interval)
    if dstat is None:
        return None
    result = _get_result_for_run(run, dataset_name, session)
    result['dstat'] = dstat
    return result


def get_subunit_results(build_uuid, dataset_name, sample_interval, db_uri,
                        build_name='tempest-full'):
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
        result = _get_data_for_run(run, dataset_name, sample_interval, session)
        if result:
            results.append(result)
    session.close()
    return results


def get_subunit_results_for_run(run, dataset_name, sample_interval,
                                db_uri=None):
    if db_uri:
        # When running from a local set the db_uri is not going to be set
        engine = create_engine(db_uri)
        Session = sessionmaker(bind=engine)
        session = Session()
    else:
        session = None
    return [_get_data_for_run(run, dataset_name, sample_interval, session)]


def get_runs_by_name(db_uri, build_name):
    engine = create_engine(db_uri)
    Session = sessionmaker(bind=engine)
    session = Session()
    runs = api.get_runs_by_key_value('build_name', build_name, session=session)
    return runs
