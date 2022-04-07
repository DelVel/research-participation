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
from torch import nn


class DETR(nn.Module):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DETR Config")
        parser.add_argument('--detr_no_pretrained', action='store_false')
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self._model = torch.hub.load('facebookresearch/detr:main',
                                     'detr_resnet50',
                                     pretrained=args.detr_no_pretrained)

    def forward(self, x):
        """Pass forward to the DETR model.

        :param x: An image tensor of shape (N, C, H, W)
        :return: A dictionary {'pred_logits': logits, 'pred_boxes': boxes}
            logits have shape (N, 100, 92)
            boxes have shape (N, 100, 4)
        """
        return self._model(x)
