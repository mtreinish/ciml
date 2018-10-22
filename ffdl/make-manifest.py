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

import click
import jinja2
import os
import sys

@click.command()
@click.option('--experiment', default='experiment',
              help="Name of the experiment")
@click.option('--dataset', default="dataset",
              help="Name of the dataset folder.")
@click.option('--gpus', default=0,
              help="Number of GPUs to use.")
@click.option('--s3-access-key-id', default=None, help='S3 Access Key ID')
@click.option('--s3-secret-access-key', default=None,
              help='S3 Secret Access Key')
@click.option('--s3-auth-url',
              default='https://s3.eu-geo.objectstorage.softlayer.net',
              help='Endpoint URL for the s3 storage')
@click.option('--input-bucket', default="cimldatasets",
              help="Name of the input bucket.")
@click.option('--output-bucket', default="cimloutput",
              help="Name of the input bucket.")
def make_manifest(experiment, dataset, gpus, s3_access_key_id,
                  s3_secret_access_key, s3_auth_url, input_bucket,
                  output_bucket):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_loader = jinja2.FileSystemLoader(dir_path)
    env = jinja2.Environment(loader=file_loader)
    template = env.get_template('manifest.yaml.j2')
    #Add the varibles
    output = template.render(
        experiment=experiment, dataset=dataset, gpus=gpus,
        s3_auth_url=s3_auth_url,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        input_bucket=input_bucket,
        output_bucket=output_bucket)
    print(output)

if __name__ == "__main__":
    sys.exit(make_manifest())
