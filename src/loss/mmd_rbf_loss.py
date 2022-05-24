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

import torch


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


def mmd_rbf_loss(x, y, gamma=None):
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
