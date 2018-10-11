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
import six
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore")


import boto3
import click
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from subunit2sql.db import api


default_db_uri = ('mysql+pymysql://query:query@logstash.openstack.org/'
                  'subunit2sql')

now = datetime.datetime.utcnow()


def get_s3_client(s3_profile=None, s3_url=None, s3_access_key_id=None,
                  s3_secret_access_key=None):
    """Get an s3 client by different means

    If a profile name is specified, it's used. Else we look for access_key
    ID and secret. If a URL is passed it's used, else the default AWS is used.
    """
    session_kwargs = {}
    client_kwargs = {}
    if s3_profile:
        session_kwargs['profile_name'] = s3_profile
    elif s3_access_key_id and s3_secret_access_key:
        session_kwargs['aws_access_key_id'] = s3_access_key_id
        session_kwargs['aws_secret_access_key'] = s3_secret_access_key
    if s3_url:
        client_kwargs['endpoint_url'] = s3_url
    session = boto3.Session(**session_kwargs)
    return session.client('s3', **client_kwargs)


def get_data_path(data_path=None, s3=None):
    """Data path is a string"""
    if not data_path:
        try:
            return [os.path.dirname(os.path.realpath(__file__)),
                    os.pardir, 'data']
        except NameError:
            # Running an interactive python, __file__ is not defined
            return tempfile.mkdtemp(prefix='ciml').split(os.sep)
    # This probably won't work on Windows
    data_path_list = data_path.split(os.sep)
    if data_path_list[0] == 's3:':
        try:
            s3.create_bucket(Bucket=data_path_list[2])
        except s3.exceptions.BucketAlreadyExists:
            # Bucket already there, just continue
            pass
    elif data_path_list[0] != "":
        # relative path, add base dir in front
        data_path_list = root_list + data_path_list
    else:
        os.makedirs(os.sep.join(data_path_list), exist_ok=True)
    return data_path_list


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
    out = pd.read_csv(input_io, skiprows=6).set_index('time')
    out.index = [_parse_dstat_date(x) for x in out.index]
    out.index = pd.DatetimeIndex(out.index)
    if sample_interval:
        out = out.resample(sample_interval).mean()
    # Remove any NaN from resampling
    out = out.dropna()
    return out


def _get_dstat_file(artifact_link, run_uuid=None, sample_interval=None,
                    use_http=True, data_path=None, s3=None):
    """Obtains and parses a dstat file to a pd.DatetimeIndex

    Finds a dstat file in the local cache or downloads it from the
    artifacts link, then parses it and resamples it into a
    pd.DatetimeIndex.
    """
    paths = ['controller/logs/dstat-csv_log.txt.gz',
             'controller/logs/dstat-csv_log.txt',
             'logs/dstat-csv_log.txt',
             'logs/dstat-csv_log.txt.gz']
    raw_data_folder = get_data_path(data_path=data_path, s3=s3)
    raw_data_folder.append('.raw')
    use_s3 = (raw_data_folder[0] == 's3:')
    stream_or_file = False
    # If s3 is None, get a vanilla s3 client just for the exceptions
    s3 = s3 or get_s3_client()
    # Check if the data is cached locally or on s3
    if not use_s3:
        os.makedirs(os.sep.join(raw_data_folder), exist_ok=True)
        raw_data_file = os.sep.join(raw_data_folder + [run_uuid + '.csv.gz'])
        # If a cache is found, use it
        if os.path.isfile(raw_data_file):
            stream_or_file = lambda: raw_data_file
            data_cleaner = lambda: os.remove(raw_data_file)
    else:
        object_key = os.sep.join(raw_data_folder[3:] + [run_uuid + '.csv.gz'])
        try:
            s3.head_object(Bucket=raw_data_folder[2], Key=object_key)
            stream_or_file = lambda: s3.get_object(
                Bucket=raw_data_folder[2], Key=object_key)['Body']
            data_cleaner = lambda: s3.delete_object(
                Bucket=raw_data_folder[2], Key=object_key)
        except s3.exceptions.ClientError:
            # Not found, continue
            pass

    # If cached
    if stream_or_file:
        try:
            with gzip.open(stream_or_file(), mode='r') as f:
                try:
                    if use_http:
                        # When using remote let me know if loading from cache
                        print("%s: dstat found in cache" % run_uuid)
                    return _parse_dstat_file(f, sample_interval)
                except pd.errors.ParserError:
                    print('Currupted data in %s, deleting.' % raw_data_file,
                          file=sys.stderr)
                    os.remove(raw_data_file)
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load this run.
            print('Run %s found in the dataset, however: %s',
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
        if use_s3:
            s3.put_object(Bucket=raw_data_folder[2], Key=object_key,
                          Body=gzip.compress(resp.text.encode()))
        else:
            with gzip.open(raw_data_file, mode='wb') as local_cache:
                local_cache.write(resp.text.encode())
        print("%s: dstat cached from URL" % run_uuid)
        # And return the parse dstat
        f = io.StringIO(resp.text)
        try:
            return _parse_dstat_file(f, sample_interval)
        except pd.errors.ParserError:
            print('Failed parsing dstat data in %s' % artifact_link,
                  file=sys.stderr)
            return None
    else:
        print("%s: dstat miss from URL" % run_uuid)
        return None


def _get_result_for_run(run, session, use_db=True, get_tests=False,
                        data_path=None, s3=None):
    # First try to get the data from disk
    metadata_folder = get_data_path(data_path=data_path, s3=s3)
    metadata_folder.append('.metadata')
    use_s3 = (metadata_folder[0] == 's3:')
    stream_or_file = False
    # If s3 is None, get a vanilla s3 client just for the exceptions
    s3 = s3 or get_s3_client()
    if use_s3:
        object_key = os.sep.join(metadata_folder[3:] + [run.uuid + '.json.gz'])
        try:
            s3.head_object(Bucket=metadata_folder[2], Key=object_key)
            stream_or_file = lambda: s3.get_object(
                Bucket=metadata_folder[2], Key=object_key)['Body']
        except s3.exceptions.ClientError:
            # Not found, continue
            pass
    else:
        os.makedirs(os.sep.join(metadata_folder), exist_ok=True)
        result_file = os.sep.join(metadata_folder + [run.uuid + '.json.gz'])
        if os.path.isfile(result_file):
            stream_or_file = lambda: result_file

    # If cached
    if stream_or_file:
        try:
            with gzip.open(stream_or_file(), mode='r') as f:
                if use_db:
                    # When using remote let me know if loading from cache
                    print("%s: metadata found in cache" % run.uuid)
                return json.loads(f.read())
        except IOError as ioe:
            # Something went wrong opening the file, so we won't load
            # this run.
            print('Run %s found in the dataset, however: %s',
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
    if use_s3:
        s3.put_object(Bucket=metadata_folder[2], Key=object_key,
                      Body=gzip.compress(json.dumps(result).encode()))
    else:
        with gzip.open(result_file, mode='wb') as local_cache:
            local_cache.write(json.dumps(result).encode())
    print("%s: metadata cached from URL" % run.uuid)

    # Adding the tests after caching
    if get_tests:
        result['tests'] = tests
    return result


def _get_data_for_run(run, sample_interval, session=None,
                      use_remote=True, data_path=None, s3=None):
    # First ensure we can get dstat data
    dstat = _get_dstat_file(run.artifacts, run.uuid, sample_interval,
                            use_http=use_remote, data_path=data_path, s3=s3)
    if dstat is None:
        return None
    result = _get_result_for_run(run, session, use_db=use_remote,
                                 data_path=data_path, s3=s3)
    result['dstat'] = dstat
    return result

def _get_cached_data_for_run_id(run_id, sample_interval, data_path=None,
                                s3=None):
    Run = collections.namedtuple('Run', ['uuid', 'artifacts'])
    return _get_data_for_run(Run(uuid=run_id, artifacts=None), sample_interval,
                             session=None, use_remote=False,
                             data_path=data_path, s3=s3)

def get_subunit_results(build_uuid, dataset_name, sample_interval, db_uri,
                        build_name='tempest-full', data_path=None, s3=None):
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
        result = _get_data_for_run(run, sample_interval, session,
                                   data_path=data_path, s3=s3)
        if result:
            results.append(result)
    session.close()
    return results


def get_subunit_results_for_run(run_id, sample_interval, data_path=None,
                                s3=None):
    """Get data for run from cache only"""
    return _get_cached_data_for_run_id(run_id, sample_interval,
                                      data_path=data_path, s3=s3)

def gather_and_cache_results_for_runs(runs, build_name, limit, sample_interval,
                                      db_uri=None, session=None, data_path=None,
                                      s3=None):
    # Download and cache dstat and metadata for a list of runs
    no_data_runs = set(
        load_run_uuids(".raw", name="unavailable",
                       data_path=data_path, s3=s3) or [])
    data_runs = set(
        load_run_uuids(".raw", name=build_name,
                       data_path=data_path, s3=s3) or [])
    # This allows re-using a session
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
            save_run_uuids(".raw", no_data_runs, name="unavailable",
                           data_path=data_path, s3=s3)
            save_run_uuids(".raw", data_runs, name=build_name,
                           data_path=data_path, s3=s3)
            print("Check-point, saved to storage %d" % len(data_runs))
        if len(data_runs) == limit and limit != 0:
            return data_runs
        if run.uuid in no_data_runs:
            print("%s: ignored by configuration" % run.uuid)
            continue
        if run.uuid in data_runs:
            print("%s: already cached" % run.uuid)
            continue
        result = _get_data_for_run(run, sample_interval, session,
                                   data_path=data_path, s3=s3)
        if result:
            data_runs.add(run.uuid)
            print('%d[%s]: Data found' % (count, run.uuid))
        else:
            no_data_runs.add(run.uuid)
            print('%d[%s]: No data' % (count, run.uuid))
    save_run_uuids(".raw", no_data_runs, name="unavailable",
                   data_path=data_path, s3=s3)
    save_run_uuids(".raw", data_runs, name=build_name,
                   data_path=data_path, s3=s3)
    print("Stored %d run IDs in .raw for build name %s" % (
        len(data_runs), build_name))
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


def get_data_json_folder_list(dataset, sub_folder=None, data_path=None,
                              s3=None):
    dataset_folder = get_data_path(data_path=data_path, s3=s3)
    dataset_folder.append(dataset)
    if sub_folder:
        dataset_folder.append(sub_folder)
    return dataset_folder

def get_data_json_folder(dataset, sub_folder=None, data_path=None,
                         s3=None):
    return os.sep.join(get_data_json_folder_list(
        dataset, sub_folder=sub_folder, data_path=data_path, s3=s3))

def save_data_json(dataset, data, name, sub_folder=None, data_path=None,
                   s3=None):
    """Save a JSON serializable object to disk"""

    # Courtesy of [fangyh]
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)

    dataset_folder = get_data_json_folder_list(dataset,
                                               sub_folder=sub_folder,
                                               data_path=data_path, s3=s3)

    # If it's a file, ensure the containing subfolder exists
    if dataset_folder[0] != 's3:':
        os.makedirs(os.sep.join(dataset_folder), exist_ok=True)
    dataset_folder.append(name + '.json.gz')

    serialized_data = json.dumps(data, cls=MyEncoder).encode()
    # For s3 the bucket is the string after s3://
    if dataset_folder[0] == 's3:':
        if not s3:
            raise ValueError("data_path %s requires an s3 client" % data_path)
        object_key = os.sep.join(dataset_folder[3:])
        s3.put_object(Bucket=dataset_folder[2], Key=object_key,
                      Body=gzip.compress(serialized_data))
    else:
        full_filename = os.sep.join(dataset_folder)
        with gzip.open(full_filename, mode='wb') as local_cache:
            local_cache.write(serialized_data)


def load_data_json(dataset, name, ignore_error=False, sub_folder=None,
                   data_path=None, s3=None):
    """Load a JSON serializable object from disk"""
    # If s3 is None, get a vanilla s3 client just for the exceptions
    s3_safe = s3 or get_s3_client()
    dataset_folder = get_data_json_folder_list(dataset,
                                               sub_folder=sub_folder,
                                               data_path=data_path, s3=s3)
    dataset_folder.append(name + '.json.gz')
    # For s3 the bucket is the string after s3://
    if dataset_folder[0] == 's3:':
        if not s3:
            raise ValueError("data_path %s requires an s3 client" % data_path)
        object_key = os.sep.join(dataset_folder[3:])
        stream_or_file = lambda: s3.get_object(Bucket=dataset_folder[2],
                                               Key=object_key)['Body']
    else:
        stream_or_file = lambda: os.sep.join(dataset_folder)
    # Unzip object stream or local file
    try:
        with gzip.open(stream_or_file(), mode='r') as f:
            return json.load(f)
    except s3_safe.exceptions.NoSuchKey:
        return None
    except FileNotFoundError:
        return None
    except IOError as ioe:
        # Ignore error
        if ignore_error:
            print('Ignore error when loading: %s', ioe)
            return None
        else:
            raise


def load_model_config(dataset, data_path=None, s3=None):
    return load_data_json(dataset, 'model_config', data_path=data_path,
                          s3=s3)


def save_model_config(dataset, model_config, data_path=None, s3=None):
    save_data_json(dataset, model_config, 'model_config', data_path=data_path,
                   s3=s3)


def load_run_uuids(dataset, name='runs', data_path=None, s3=None):
    """Return a list of run uuids for a specific dataset_name

    Read the list of run uuids from file and return a list of run uuids.
    """
    return load_data_json(dataset, name, data_path=data_path, s3=s3)


def save_run_uuids(dataset, run_uuids, name='runs', data_path=None, s3=None):
    save_data_json(dataset, list(run_uuids), name, data_path=data_path, s3=s3)


def load_experiment(name, data_path=None, s3=None):
    return load_data_json('_experiments', 'experiment', sub_folder=name,
                          data_path=data_path, s3=s3)


def save_experiment(experiment, name, data_path=None, s3=None):
    save_data_json('_experiments', experiment, 'experiment', sub_folder=name,
                   data_path=data_path, s3=s3)


def get_model_folder(dataset, name, data_path=None):
    return get_data_json_folder(dataset, sub_folder=name, data_path=data_path)


def load_dataset(dataset, name, data_path=None, s3=None):
    dataset_file = get_data_path(data_path=data_path, s3=s3)
    dataset_file.append(dataset)
    use_s3 = (dataset_file[0] == 's3:')

    if use_s3:
        if not s3:
            raise ValueError("data_path %s requires an s3 client" % data_path)

        # Download as a file first. Not that efficient, but we avoid having
        # to keep the whole dataset in memory.
        object_key = os.sep.join(dataset_file[3:] + [name + '.npz'])
        _, target_filename = tempfile.mkstemp(prefix=None, suffix='.npz')
        s3.download_file(Bucket=dataset_file[2], Key=object_key,
                         Filename=target_filename)
    else:
        target_filename = os.sep.join(dataset_file + [name + '.npz'])

    with np.load(target_filename) as numpy_dataset:
        result = {f: numpy_dataset[f] for f in numpy_dataset.files}

    if use_s3:
        os.remove(target_filename)

    return result


def save_dataset(dataset, name, data_path=None, s3=None, **kwargs):
    dataset_file = get_data_path(data_path=data_path, s3=s3)
    dataset_file.append(dataset)
    use_s3 = (dataset_file[0] == 's3:')

    if use_s3:
        if not s3:
            raise ValueError("data_path %s requires an s3 client" % data_path)
        _, target_filename = tempfile.mkstemp(prefix=None, suffix='.npz')
    else:
        target_filename = os.sep.join(dataset_file + [name + '.npz'])
    np.savez_compressed(target_filename, **kwargs)

    if use_s3:
        object_key = os.sep.join(dataset_file[3:] + [name + '.npz'])
        s3.upload_file(Filename=target_filename, Bucket=dataset_file[2],
                       Key=object_key)
        # Remove the tmp file
        print(target_filename)
        os.remove(target_filename)


@click.command()
@click.option('--dataset-experiment-label', nargs=3, type=str,
              help="Name of the dataset and experiment.", multiple=True)
@click.option('--dataset-experiment-comp', nargs=2, type=str,
              help="Name of the dataset and experiment to compare.",
              multiple=True)
@click.option('--experiment-sets-names', nargs=2, type=str,
              help="Name of the experiment sets o compare.")
@click.option('--data-keys', '-k', multiple=True, help="Key to export")
@click.option('--data-path', default=None,
              help="Path to the raw data, local path or s3://<bucket>")
@click.option('--s3-profile', default=None, help='Named configuration')
@click.option('--s3-url',
              default='https://s3.eu-geo.objectstorage.softlayer.net',
              help='Endpoint URL for the s3 storage')
@click.option('--output', help="Name of the output file")
@click.option('--title', help="Title of the graph")
def plot_experiment_data(dataset_experiment_label,
                         dataset_experiment_comp, experiment_sets_names,
                         data_keys, data_path,
                         s3_profile, s3_url, output, title):
    # Do some input validation when dataset_experiment_label_comp is set
    if dataset_experiment_comp:
        if not experiment_sets_names:
            print("Please provide the name for the two experiment sets")
            sys.exit(1)
        d1_len = len(dataset_experiment_label)
        d2_len = len(dataset_experiment_comp)
        if d1_len != d2_len:
            print("The experiments to compare must have the same size")
            print("Dataset1: %d, dataset2 %d" % (d1_len, d2_len))
            sys.exit(1)
        if len(data_keys) > 1:
            print("Cannot have two experiments and multiple keys")
            sys.exit(1)

    s3 = get_s3_client(s3_profile=s3_profile, s3_url=s3_url)
    # Define the array size
    array_size = 2 if dataset_experiment_comp else len(data_keys)
    data = np.ndarray(shape=(len(dataset_experiment_label), array_size))
    labels = []
    for count, d_e_l in enumerate(dataset_experiment_label):
        exp_data = load_data_json(d_e_l[0], 'eval_' + d_e_l[0],
                                  sub_folder=d_e_l[1],
                                  data_path=data_path, s3=s3)
        labels.append(d_e_l[2])
        if dataset_experiment_comp:
            d2, e2 = dataset_experiment_comp[count]
            exp_data_comp = load_data_json(d2, 'eval_' + d2,
                                           sub_folder=e2,
                                           data_path=data_path, s3=s3)
        data_count = []
        for count_k, k in enumerate(data_keys):
            if k == 'accuracy':
                data_count.append(1 - exp_data[k])
                if dataset_experiment_comp:
                    data_count.append(1 - exp_data_comp[k])
            else:
                data_count.append(exp_data[k])
                if dataset_experiment_comp:
                    data_count.append(exp_data_comp[k])

        data[count] = data_count

    # Setup the dataframe
    if not dataset_experiment_comp:
        columns = data_keys
    else:
        columns = experiment_sets_names
    df = pd.DataFrame(data, index=labels, columns=columns)
    print("Plotting %s with data %s" % (title, df))
    use_legends = (len(columns) > 1)
    data_plot = df.plot(kind='bar', title=title, legend=use_legends, rot=0)
    data_plot.get_figure().savefig(output)


@click.command()
@click.option('--build-name', default="tempest-full", help="Build name.")
@click.option('--db-uri', default=default_db_uri, help="DB URI")
@click.option('--limit', default=0, help="Maximum number of entries")
@click.option('--data-path', default=None,
              help="Path to the data, local or in the s3://<bucket> format")
@click.option('--s3-profile', default=None, help='Named configuration')
@click.option('--s3-url',
              default='https://s3.eu-geo.objectstorage.softlayer.net',
              help='Endpoint URL for the s3 storage')
def cache_data(build_name, db_uri, limit, data_path, s3_profile, s3_url):
    cache_data_function(build_name, db_uri, limit, data_path, s3_profile,
                        s3_url)


def cache_data_function(build_name, db_uri, limit=0, data_path=None,
                        s3_profile=None, s3_url=None, s3_access_key_id=None,
                        s3_secret_access_key=None):
    runs = get_runs_by_name(db_uri, build_name=build_name)
    print("Obtained %d runs named %s from the DB" % (len(runs), build_name))
    s3=get_s3_client(s3_url=s3_url,s3_profile=s3_profile,
                     s3_access_key_id=s3_access_key_id,
                     s3_secret_access_key=s3_secret_access_key)
    limit_runs = gather_and_cache_results_for_runs(
        runs, build_name, limit, '1s', db_uri, data_path=data_path, s3=s3)
    return {
        'build_name': build_name,
        'num_examples': len(limit_runs),
        'data_path': data_path,
        's3_url': s3_url
    }


def main(args):
    """Main function for invocation via action"""
    if 'payload' in args:
        del args['payload']
    try:
        cache_data_function(db_uri=default_db_uri, limit=0, **args)
    except Exception as e:
        print(e)
    finally:
        # Ensure we always return a dict, even on failure
        return args
