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
from pytorch_metric_learning.distances import BaseDistance

from src.functional import smooth_chamfer_matching


class ChamferSimilarity(BaseDistance):
    @classmethod
    def add_module_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("ChamferSimilarity")
        parser.add_argument('--sim_temp', type=float, default=1.0)
        return parent_parser

    def __init__(self, args):
        super().__init__(is_inverted=True)
        self.temp = args.sim_temp

    def compute_mat(self, query_emb, ref_emb):
        return smooth_chamfer_matching(query_emb, ref_emb, self.temp)

    def pairwise_distance(self, query_emb, ref_emb):
        return smooth_chamfer_matching(query_emb, ref_emb,
                                       self.temp).diagonal()
