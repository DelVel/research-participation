#  Copyright 2022 https://github.com/KevinMusgrave/pytorch-metric-learning
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

import torch
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from src.third_party.base_miner import BaseTupleMiner


class TripletMarginMiner(BaseTupleMiner):
    """
    Returns triplets that violate the margin
    Args:
        margin
        type_of_triplets: options are "all", "hard", or "semihard".
                "all" means all triplets that violate the margin
                "hard" is a subset of "all", but the negative is closer to the anchor than the positive
                "semihard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, margin=0.2, type_of_triplets="all", **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(list_of_names=["margin"],
                                          is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist",
                           "neg_pair_dist"],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        anchor_idx, positive_idx, negative_idx = lmu.get_all_triplets_indices(
            labels, ref_labels
        )
        mat = self.distance(embeddings, ref_emb)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist
        )

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semihard":
                threshold_condition &= triplet_margin > 0

        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )

    def set_stats(self, ap_dist, an_dist, triplet_margin):
        if self.collect_stats:
            with torch.no_grad():
                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()
