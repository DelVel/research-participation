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
from einops import repeat
from torch import Tensor


def diversity_loss(x: Tensor) -> Tensor:
    """
    Measures diversity of the input.

    :param x: A tensor of shape [B x S x D]
    :return: A scalar tensor
    """
    assert x.dim() == 3
    batch, seq_len, seq_dim = x.shape
    x = torch.nn.functional.normalize(x, dim=2)
    x = torch.bmm(x, x.transpose(1, 2))
    identity = torch.eye(seq_len, device=x.device, dtype=torch.bool)
    identity = repeat(identity, '... -> b ...', b=batch)
    x = x.masked_fill(identity, 0)
    b_loss = x.norm(p=2, dim=(1, 2)) / (seq_len ** 2)
    return b_loss.mean()
