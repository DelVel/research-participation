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
import torch.nn.functional as f
from einops import rearrange
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from torch import nn

from src.functional import smooth_chamfer_matching


class ChamferSimilarity:
    def __init__(self, temp=1.0):
        self.temp = temp

    def __call__(self, query_emb, ref_emb):
        query_emb = f.normalize(query_emb, dim=-1)
        ref_emb = f.normalize(ref_emb, dim=-1)
        return smooth_chamfer_matching(query_emb, ref_emb, self.temp)


class ChamferTripletLoss(nn.Module):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ChamferTripletLoss")
        parser.add_argument('--loss_margin', type=float, default=0.05)
        parser.add_argument('--loss_temperature', type=float, default=1.0)
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.margin = args.loss_margin
        self.similarity = ChamferSimilarity(temp=args.loss_temperature)
        self.reducer = AvgNonZeroReducer()

    def forward(self, img_emb, txt_emb):
        b_size = img_emb.shape[0]
        t_size = txt_emb.shape[1]
        txt_emb = rearrange(txt_emb, 'b t ... -> (b t) 1 ...')
        img_label = torch.arange(b_size).to(img_emb.device)
        txt_label = torch.arange(b_size).repeat_interleave(t_size).to(
            img_emb.device)
        i2t_loss = self.compute_loss(img_emb, img_label, txt_emb, txt_label)
        t2i_loss = self.compute_loss(txt_emb, txt_label, img_emb, img_label)
        return (i2t_loss + t2i_loss) / 2

    def compute_loss(self, embeddings, labels, ref_emb, ref_labels):
        indices_tuple = lmu.convert_to_triplets(
            None, labels, ref_labels,
            t_per_anchor='all'
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.similarity(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        current_margins = an_dists - ap_dists
        violation = current_margins + self.margin
        loss = f.relu(violation)
        loss_dict = {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }
        reducer = self.reducer(loss_dict, embeddings, labels)
        return reducer
