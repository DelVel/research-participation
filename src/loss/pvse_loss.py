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

from torch import nn

from src.loss.diversity import diversity_loss
from src.loss.mmd_rbf_loss import mmd_rbf_loss
from src.loss.triplet_loss import simple_triplet_loss


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

    def forward(self, x, y, xs, ys):
        img_res = xs[1]
        txt_res = ys[1]
        triplet_loss = simple_triplet_loss(self.similarity(x, y)).mean()
        div_loss = self.div_constant * (
                diversity_loss(img_res) + diversity_loss(txt_res))
        mmd_loss = self.mmd_constant * (
            mmd_rbf_loss(img_res, txt_res))
        return triplet_loss + div_loss + mmd_loss
