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

from torch import autograd
import torch.nn as nn
from torch.utils import data

class DstatDataset(data.Dataset):
    def __init__(self, dstat_data, transform=None):
        self.dstat_data = dstat_data

    def __len__(self):
        return len(self.dstat_data)

    def __getitem__(self, idx):
        stats = self.dstat_data.iloc[idx, 1:].as_matrix()
        if self.transform:
            stats = self.transform(stats)
        return stats

class DstatRNN(nn.modules):
    def __init__(self, input_size, hidden_size, output_size):
        pass


def train(rnn, category_tensor, line_tensor):
    pass
