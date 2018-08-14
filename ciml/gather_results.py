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

import collections
import datetime
import gzip
import io
import itertools
import json
import os
import sys

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


def _get_dstat_file(artifact_link, run_uuid=None, sample_interval=None,
                    use_http=True, data_path=None):
    """Obtains and parses a dstat file to a pandas.DatetimeIndex

    Finds a dstat file in the local cache or downloads it from the
    artifacts link, then parses it and resamples it into a
    pandas.DatetimeIndex.
    """
    paths = ['controller/logs/dstat-csv_log.txt.gz',
             'controller/logs/dstat-csv_log.txt',
             'logs/dstat-csv_log.txt',
             'logs/dstat-csv_log.txt.gz']
    if not data_path:
        raw_data_folder = [os.path.dirname(os.path.realpath(__file__)),
                           os.pardir, 'data', '.raw']
    else:
        raw_data_folder = [data_path, 'data', '.raw']
    os.makedirs(os.sep.join(raw_data_folder), exist_ok=True)
    raw_data_file = os.sep.join(raw_data_folder + [run_uuid + '.csv.gz'])
    if os.path.isfile(raw_data_file):
        try:
            with gzip.open(raw_data_file, mode='r') as f:
                try:
                    if use_http:
                        # When using remote let me know if loading from cache
                        print("%s: dstat found in cache" % run_uuid)
                    return _parse_dstat_file(f, sample_interval)
                except pandas.errors.ParserError:
                    print('Currupted data in %s, deleting.' % raw_data_file,
                          file=sys.stderr)
                    os.remove(raw_data_file)
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load this run.
            print('Run %s found in the local dataset, however: %s',
                  (run_uuid, ioe))
            return None

    # If no cache and use_http is False, nothing more we can do
    if not use_http:
        print("No local cache found for %s, and use_http false" % run_uuid)

    # If no local cache was found we try to fetch the dstats file via HTTP
    # and we store it in cache.
    for path in paths:
        if not artifact_link:
            break
        url = artifact_link + '/' + path
        resp = requests.get(url)
        if resp.status_code == 404:
            continue
        # Cache the file locally
        with gzip.open(raw_data_file, mode='wb') as local_cache:
            local_cache.write(resp.text.encode())
        print("%s: dstat cached from URL" % run_uuid)
        # And return the parse dstat
        f = io.StringIO(resp.text)
        try:
            return _parse_dstat_file(f, sample_interval)
        except pandas.errors.ParserError:
            print('Failed parsing dstat data in %s' % artifact_link,
                  file=sys.stderr)
            return None
    else:
        print("%s: dstat miss from URL" % run_uuid)
        return None


def _get_result_for_run(run, session, use_cache=True, use_db=True,
                        get_tests=False, data_path=None):
    # First try to get the data from disk
    if not data_path:
        metadata_folder = [os.path.dirname(os.path.realpath(__file__)),
                           os.pardir, 'data', '.metadata']
    else:
        metadata_folder = [data_path, 'data', '.metadata']
    os.makedirs(os.sep.join(metadata_folder), exist_ok=True)
    result_file = os.sep.join(metadata_folder + [run.uuid + '.json.gz'])
    if use_cache:
        if os.path.isfile(result_file):
            try:
                if use_db:
                    # When using remote let me know if loading from cache
                    print("%s: metadata found in cache" % run.uuid)
                with gzip.open(result_file, mode='r') as f:
                    return json.loads(f.read())
            except IOError as ioe:
                # Something went wrong opening the file, so we won't load
                # this run.
                print('Run %s found in the local dataset, however: %s',
                      (run.uuid, ioe))
                return None
    # If no local cache, and use_db is False, return nothing
    if not use_db:
        print("No local data for %s, use_db set to false" % run.uuid)
        return None

    # If no local cache, get data from the DB
    result = {}

    # We may need the list of tests
    if get_tests:
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

    # Setup run metadata
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
    print("%s: metada cached from URL")

    # Adding the tests after caching
    if get_tests:
        result['tests'] = tests
    return result


def _get_data_for_run(run, sample_interval, session=None, use_cache=True,
                      use_remote=True, data_path=None):
    # First ensure we can get dstat data
    dstat = _get_dstat_file(run.artifacts, run.uuid, sample_interval,
                            use_http=use_remote, data_path=data_path)
    if dstat is None:
        return None
    result = _get_result_for_run(run, session, use_cache,
                                 use_db=use_remote, data_path=data_path)
    result['dstat'] = dstat
    return result

def _get_local_data_for_run_id(run_id, sample_interval, data_path=None):
    Run = collections.namedtuple('Run', ['uuid', 'artifacts'])
    return _get_data_for_run(Run(uuid=run_id, artifacts=None), sample_interval,
                             session=None, use_cache=True, use_remote=False,
                             data_path=data_path)

def get_subunit_results(build_uuid, dataset_name, sample_interval, db_uri,
                        build_name='tempest-full', use_cache=True,
                        data_path=None):
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
        result = _get_data_for_run(run, sample_interval, session, use_cache,
                                   data_path=data_path)
        if result:
            results.append(result)
    session.close()
    return results


def get_subunit_results_for_run(run_id, sample_interval, data_path=None):
    """Get data for run from cache only"""
    return _get_local_data_for_run_id(run_id, sample_interval,
                                      data_path=data_path)

def gather_and_cache_results_for_runs(runs, limit, sample_interval, db_uri=None,
                                      session=None, data_path=None):
    # Download and cache dstat and metadata for a list of runs
    # This allows re-using a session
    no_data_runs = set(
        load_run_uuids(".unavailable", data_path=data_path) or [])
    data_runs = set([])
    if db_uri:
        # When running from a local set the db_uri is not going to be set
        engine = create_engine(db_uri)
        Session = sessionmaker(bind=engine)
        session = Session()
    else:
        session = session
    for count, run in enumerate(runs):
        if count % 100 == 0:
            # Every 100 runs save to disk so we can restore interrupted jobs.
            save_run_uuids(".unavailable", no_data_runs, data_path=data_path)
        if len(data_runs) == limit:
            return data_runs
        if run.uuid in no_data_runs:
            print("%s: ignored by configuration" % run.uuid)
            continue
        result = _get_data_for_run(run, sample_interval, session,
                                   use_cache=True, data_path=data_path)
        if result:
            data_runs.add(run.uuid)
            print('%d[%s]: Data found' % (count, run.uuid))
        else:
            no_data_runs.add(run.uuid)
            print('%d[%s]: No data' % (count, run.uuid))
    save_run_uuids(".unavailable", no_data_runs, data_path=data_path)
    return data_runs

def get_runs_by_name(db_uri, build_name, session=None):
    if not session:
        engine = create_engine(db_uri)
        Session = sessionmaker(bind=engine)
        session = Session()
    runs = api.get_runs_by_key_value('build_name', build_name, session=session)
    fail = [x for x in runs if x.fails > 0]
    passes = [x for x in runs if x.fails == 0]
    out = list(itertools.chain.from_iterable(zip(fail, passes)))
    leftover = (len(out) / 2) + 1
    full_out = out + passes[int(leftover):]
    return full_out


def save_model_config(dataset, model_config, data_path=None):
    if not data_path:
        data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                       'data', dataset]
    else:
        data_folder = [data_path, 'data', dataset]
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


def load_model_config(dataset, data_path=None):
    if not data_path:
        data_folder = [os.path.dirname(os.path.realpath(__file__)), os.pardir,
                       'data', dataset]
    else:
        data_folder = [data_path, 'data', dataset]
    model_config_file = os.sep.join(data_folder + [dataset + '.json.gz'])
    if os.path.isfile(model_config_file):
        try:
            with gzip.open(model_config_file, mode='r') as f:
                return json.loads(f.read())
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load this run.
            print('Dataset config found in the local dataset, however: %s',
                  ioe)
            return None


def load_run_uuids(dataset, data_path=None):
    """Return a list of run uuids for a specific dataset_name

    Read the list of run uuids from file and return a list of run uuids.
    """

    if not data_path:
        dataset_runs = os.sep.join([
            os.path.dirname(os.path.realpath(__file__)), os.pardir, 'data',
            dataset, 'runs.json.gz'])
    else:
        dataset_runs = os.sep.join([data_path, 'data', dataset,
                                    'runs.json.gz'])
    if os.path.isfile(dataset_runs):
        try:
            with gzip.open(dataset_runs, mode='r') as f:
                return json.load(f)
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load this run.
            print('Run found in the local dataset, however: %s', ioe)
            return None


def save_run_uuids(dataset, run_uuids, data_path=None):
    if not data_path:
        dataset_folder = [os.path.dirname(os.path.realpath(__file__)),
                          os.pardir]
    else:
        dataset_folder = [data_path]
    dataset_folder.extend(['data', dataset])
    os.makedirs(os.sep.join(dataset_folder), exist_ok=True)
    dataset_folder.append('runs.json.gz')
    dataset_runs = os.sep.join(dataset_folder)
    with gzip.open(dataset_runs, mode='wb') as local_cache:
        local_cache.write(json.dumps(list(run_uuids)).encode())
