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

from ciml import dstat_data
from ciml import gather_results
from ciml import listener


def main():
    event_queue = queue.Queue()
    listen_thread = listener.MQTTSubscribe(event_queue,
                                           'firehose.openstack.org',
                                           'gearman-subunit/#')
    listen_thread.start()
    dstat_model = dstat_data.DstatTrainer()
    while True:
        event = event_queue.get()
        results = gather_results.get_subunit_results(
            event['build_uuid'],
            'mysql+pymysql://query:query@logstash.openstack.org/subunit2sql')
        for result in results:
            dstat_model.train(result['dstat'], result['status'])


if __name__ == "__main__":
    main()
