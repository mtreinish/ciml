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

import json
import threading

import paho.mqtt.client as mqtt


class MQTTSubscribe(threading.Thread):

    def __init__(self, output_queue, hostname, topic, port=1883,
                 websocket=False, client_id=None, keepalive=60,
                 will=None, auth=None, tls=None, qos=0):
        super(MQTTSubscribe, self).__init__()
        self.queue = output_queue
        self.hostname = hostname
        self.port = port
        self.client_id = client_id
        self.keepalive = keepalive
        self.mqtt_topic = topic
        self.will = will
        self.auth = auth
        self.tls = tls
        self.qos = qos
        transport = "tcp"
        if websocket:
            transport = "websocket"
        self.client = mqtt.Client(transport=transport)
        if tls:
            self.client.tls_set(**tls)
        if auth:
            self.client.username_pw_set(auth['username'],
                                        password=auth.get('password'))

    def run(self):
        def on_connect(client, userdata, flags, rc):
            client.subscribe(self.mqtt_topic)

        def on_message(client, userdata, msg):
            output = json.loads(msg.payload)
            self.queue.put(output)

        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.connect(self.hostname, self.port)
        self.client.loop_forever()
