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
from torch import Tensor


def lse(x: Tensor, dim, keep_dim=False, temp=1.0):
    return (temp * x).logsumexp(dim=dim, keepdim=keep_dim) / temp


def smooth_chamfer_matching(x: Tensor, y: Tensor, temp=1.0):
    """
    Smooth Chamfer Matching.

    :param temp: Temperature
    :param x: (*1, M, D), normalized D.
    :param y: (*2, N, D), normalized D.
    :return: (*1, *2)
    """
    x_dim = x.dim()
    y_dim = y.dim()
    inner: Tensor = torch.inner(x, y)
    ind_m = x_dim - 2
    ind_n = x_dim + y_dim - 3
    x_side: Tensor = lse(inner, dim=ind_m, temp=temp)
    x_mean: Tensor = x_side.mean(dim=ind_n - 1)
    y_side: Tensor = lse(inner, dim=ind_n, temp=temp)
    y_mean: Tensor = y_side.mean(dim=ind_m)
    return (x_mean + y_mean) / 2
