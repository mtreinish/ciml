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

import queue
import sys

import click

from ciml import gather_results
from ciml import listener
from ciml import svm_trainer
from ciml import trainer


default_mqtt_hostname = ('firehose.openstack.org')
default_db_uri = ('mysql+pymysql://query:query@logstash.openstack.org/'
                  'subunit2sql')

@click.command()
@click.option('--db-uri', default=default_db_uri, help="DB URI")
@click.option('--mqtt-hostname', default=default_mqtt_hostname,
              help='MQTT hostname')
@click.option('--topic', default='gearman-subunit/#',
              help='MQTT topic to subscribe to')
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--sample-interval', default='1s',
              help='dstat (down)sampling interval')
@click.option('--build-name', default="tempest-full", help="Build name.")
def mqtt_predict(db_uri, mqtt_hostname, topic, dataset, sample_interval,
                 build_name):
    event_queue = queue.Queue()
    listen_thread = listener.MQTTSubscribe(event_queue, mqtt_hostname, topic)

    while True:
        event = event_queue.get()
        results = gather_results.get_subunit_results(
            event['build_uuid'], dataset, sample_interval, db_uri, build_name)
        run_uuids = [r['uuid'] for r in results]
        for res in results:
            vector, status, labels = trainer.normalize_example(res)
            model = svm_trainer.SVMTrainer(
                vector, [event['build_uuid']]*len(results), labels, [status],
                dataset_name=dataset) 
