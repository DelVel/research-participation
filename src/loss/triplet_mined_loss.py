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

from src.loss import TripletLoss
from src.third_party import TripletMarginMiner


class TripletMinedLoss(TripletLoss):
    @staticmethod
    def parser_hook(parser):
        parser.add_argument('--loss_mine_type', type=str, default='hard')

    def __init__(self, args, similarity):
        super().__init__(args, similarity)
        self.miner = TripletMarginMiner(distance=similarity,
                                        margin=self.margin,
                                        type_of_triplets=args.loss_mine_type)

    def get_i2t_pair(self, img_emb, img_label, txt_emb, txt_label):
        return self.miner(img_emb, img_label, txt_emb, txt_label)

    def get_t2i_pair(self, img_emb, img_label, txt_emb, txt_label):
        return self.miner(txt_emb, txt_label, img_emb, img_label)
