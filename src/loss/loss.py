#  Copyright 2022 Taegyu Park
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

import torch
from einops import reduce, rearrange, repeat
from torch.nn import Module
from torch.nn.functional import normalize


class InnerLoss(Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.covar = None
        self.batch = None

    def forward(self, img, text):
        """
        Loss function.

        :param img: Shape of (batch, z_per_i, embedding_dim)
        :param text: Shape of (batch, captions, embedding_dim)
        :return: Loss value.
        """
        assert img.shape[0] == text.shape[0]
        assert img.shape[2] == text.shape[2]
        self._get_inner(img, text)
        self.batch = img.shape[0]
        return self.post_forward(img, text)

    @abstractmethod
    def post_forward(self, img, text):
        """
        Forward pass after getting inner product.

        :return: self.covar for inner product tensor, self.batch for batch size
        of the input.
        """
        pass

    def _get_inner(self, img, text):
        img = normalize(img, dim=2)
        text = normalize(text, dim=2)
        self.covar = torch.inner(img, text)


class CrossLoss(InnerLoss):
    def post_forward(self, img, text):
        covar = self.covar
        covar = reduce(covar, 'b1 i b2 t -> b1 b2 t', 'max')
        covar = reduce(covar, 'b1 b2 t -> b1 b2', 'mean')
        max_elem = covar.trace() / self.batch
        min_elem = 0
        if self.batch > 1:
            min_elem = torch.triu(covar, diagonal=1).sum() / (
                    self.batch * (self.batch - 1) / 2)
        return min_elem - max_elem


class ContrastiveLoss(InnerLoss):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Loss Config")
        parser.add_argument('--loss_temperature', type=float, default=1.0)
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.temperature = args.loss_temperature

    def post_forward(self, img, text):
        self.covar = torch.logsumexp(self.temperature * self.covar, dim=1)
        return self._sum_up_loss()

    def _sum_up_loss(self):
        loss = 0
        loss += self._i2t()
        loss += self._t2i()
        return loss

    def _i2t(self):
        t_dim = self.covar.shape[2]
        temp = rearrange(self.covar, 'bi bt t -> bi (bt t)')
        temp = temp.softmax(dim=1)
        temp = temp / self.temperature
        temp = repeat(temp, 'bi (bt t) -> bi bt t', t=t_dim)
        mean = reduce(temp, 'bi bt t -> bi bt', 'mean')
        loss = (mean * self._get_coefficient_mat()).sum()
        return loss

    def _t2i(self):
        temp = self.covar
        temp = temp.softmax(dim=0)
        temp = temp / self.temperature
        mean = reduce(temp, 'bi bt t -> bi bt', 'sum')
        loss = (mean * self._get_coefficient_mat()).sum()
        return loss

    def _get_coefficient_mat(self):
        b = self.batch
        tensor = torch.ones(b, b) - b * torch.eye(b)
        tensor /= b * (b - 1)
        return tensor.to(self.covar.device)
