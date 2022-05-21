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
from einops import rearrange
from torch import nn

from src.third_party import TripletMarginLoss


class TripletLoss(nn.Module):
    @classmethod
    def add_module_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--loss_margin', type=float, default=0.05)
        cls.parser_hook(parser)
        return parent_parser

    @staticmethod
    def parser_hook(parser):
        # To be overridden by subclasses
        pass

    def __init__(self, args, similarity):
        super().__init__()
        self.margin = args.loss_margin
        self.loss = TripletMarginLoss(distance=similarity,
                                      margin=self.margin)

    def forward(self, img_emb, txt_emb):
        b_size = img_emb.shape[0]
        t_size = txt_emb.shape[1]
        txt_emb = rearrange(txt_emb, 'b t ... -> (b t) 1 ...')
        img_label = torch.arange(b_size, device=img_emb.device)
        txt_label = torch.arange(b_size, device=img_emb.device) \
            .repeat_interleave(t_size)
        i_t_p = self.get_i2t_pair(img_emb, img_label, txt_emb, txt_label)
        t_i_p = self.get_t2i_pair(img_emb, img_label, txt_emb, txt_label)
        i_t_l = self.loss(img_emb, img_label, i_t_p, txt_emb, txt_label)
        t_i_l = self.loss(txt_emb, txt_label, t_i_p, img_emb, img_label)
        return (i_t_l + t_i_l) / 2

    def get_i2t_pair(self, img_emb, img_label, txt_emb, txt_label):
        return None

    def get_t2i_pair(self, img_emb, img_label, txt_emb, txt_label):
        return None
