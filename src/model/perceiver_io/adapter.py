#  Copyright 2022 https://github.com/krasserm/perceiver-io
#  Modified by Taegyu Park
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from abc import ABCMeta, abstractmethod

import torch.nn as nn


class InputAdapter(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_input_channels):
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels

    @abstractmethod
    def forward(self, x):
        """Converts the given input to the format expected by the model.

        :param x: A tensor of shape (batch_size, *).
        :return: A tuple of (converted tensor, mask tensor | None).
        """
        raise NotImplementedError()


class OutputAdapter(nn.Module, metaclass=ABCMeta):
    def __init__(self, output_shape):
        super().__init__()
        self._output_shape = output_shape

    @property
    def output_shape(self):
        return self._output_shape

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()
