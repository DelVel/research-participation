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
import math

import torch
from einops import rearrange, repeat
from torch import nn, Tensor
from torch.nn import functional as F


class PVSELoss(nn.Module):
    @classmethod
    def add_module_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        return parent_parser

    def __init__(self, args, similarity):
        super().__init__()
        self.similarity = similarity
        self.div_constant = 0.1
        self.mmd_constant = 0.01
        self.dict = dict()

    def forward(self, img, txt, xs, ys):
        img_res = xs[1]
        txt_res = ys[1]
        triplet_loss = simple_triplet_loss(self.similarity(img, txt))
        div_loss = self.div_constant * (
                diversity_loss(img_res) + diversity_loss(txt_res)) / 2
        mmd_loss = self.mmd_constant * (
            mmd_rbf_loss(img, txt))
        self.dict['triplet_loss'] = triplet_loss.item()
        self.dict['div_loss'] = div_loss.item()
        self.dict['mmd_loss'] = mmd_loss.item()
        return triplet_loss + div_loss + mmd_loss


def simple_triplet_loss(sim: Tensor):
    """
    Simple triplet loss with max policy & margin 0.1.

    :param sim: A tensor of [B x B] .
    :return: Triplet loss.
    """
    b = sim.shape[0]
    mask = torch.eye(b, device=sim.device, dtype=torch.bool)
    diagonal = sim.diag()
    i2t = (sim - rearrange(diagonal, 'b -> b 1') + 0.1).clamp(min=0)
    t2i = (sim - rearrange(diagonal, 'b -> 1 b') + 0.1).clamp(min=0)
    i2t = i2t.masked_fill(mask, 0)
    t2i = t2i.masked_fill(mask, 0)
    i2t = i2t.max(dim=1)[0]
    t2i = t2i.max(dim=0)[0]
    return ((i2t + t2i) / 2).mean()


def diversity_loss(x: Tensor) -> Tensor:
    """
    Measures diversity of the input.

    :param x: A tensor of shape [B x S x D]
    :return: A scalar tensor
    """
    assert x.dim() == 3
    batch, seq_len, seq_dim = x.shape
    x = F.normalize(x, dim=2)
    x = torch.bmm(x, x.transpose(1, 2))
    identity = torch.eye(seq_len, device=x.device, dtype=torch.bool)
    identity = repeat(identity, '... -> b ...', b=batch)
    x = x.masked_fill(identity, 0)
    b_loss = x.norm(p=2, dim=(1, 2)) / (seq_len ** 2)
    return b_loss.mean()


def _rbf(x, y, gamma):
    """
    RBF kernel.

    :param x: A tensor of shape [B x S x D]
    :param y: A tensor of shape [B x S' x D]
    :param gamma: RBF constant
    :return: A tensor of shape [B x S x S']
    """
    assert x.dim() == 3
    assert y.dim() == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[2]
    cdist = torch.cdist(x, y)
    return torch.exp(-gamma * cdist)


def mmd_rbf_loss(x, y, gamma=0.5):
    """
    MMD with RBF kernel.

    :param x: A tensor of shape [B x S x D]
    :param y: A tensor of shape [B x S' x D]
    :param gamma: RBF constant
    :return: A scalar tensor
    """
    assert x.dim() == 3
    assert y.dim() == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[2]
    if gamma is None:
        gamma = 1.0 / x.shape[-1]
    x_kernel = _rbf(x, x, gamma).mean()
    y_kernel = _rbf(y, y, gamma).mean()
    xy_kernel = _rbf(x, y, gamma).mean()
    return x_kernel + y_kernel - 2 * xy_kernel
