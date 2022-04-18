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
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from torch import nn

from src.functional import smooth_chamfer_matching


class ChamferSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)

    def compute_mat(self, query_emb, ref_emb):
        return smooth_chamfer_matching(query_emb, ref_emb)

    def pairwise_distance(self, query_emb, ref_emb):
        return smooth_chamfer_matching(query_emb, ref_emb).diagonal()


class ChamferTripletLoss(nn.Module):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ChamferTripletLoss")
        parser.add_argument('--miner_margin', type=float, default=0.2)
        parser.add_argument('--margin', type=float, default=0.05)
        parser.add_argument('--triplet_type', type=str, default='hard')
        return parent_parser

    def __init__(self, args):
        super().__init__()
        similarity = ChamferSimilarity()
        self.miner = TripletMarginMiner(
            margin=args.miner_margin,
            type_of_triplets=args.triplet_type,
            distance=similarity
        )
        self.loss = TripletMarginLoss(
            margin=args.margin,
            distance=similarity
        )

    def forward(self, img_emb, txt_emb):
        b_size = img_emb.shape[0]
        t_size = txt_emb.shape[1]
        txt_emb = rearrange(txt_emb, 'b t ... -> (b t) ...')
        img_label = torch.arange(b_size).to(img_emb.device)
        txt_label = torch.arange(b_size).repeat_interleave(t_size).to(
            img_emb.device)
        i2t_hard_pair = self.miner(img_emb, img_label, txt_emb, txt_label)
        i2t_loss = self.loss(img_emb, img_label, i2t_hard_pair, txt_emb,
                             txt_label)
        t2i_hard_pair = self.miner(txt_emb, txt_label, img_emb, img_label)
        t2i_loss = self.loss(txt_emb, txt_label, t2i_hard_pair, img_emb,
                             img_label)
        return (i2t_loss + t2i_loss) / 2
