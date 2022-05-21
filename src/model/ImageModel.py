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
import itertools

import torch
from einops import einops
from torch import nn
from torch.nn.init import xavier_normal_
from torchvision.models import resnet50
from torchvision.transforms import Compose, RandomCrop, ToTensor


class ImageTrans(nn.Module):
    seq_len = 49
    seq_dim = 2048

    @staticmethod
    def get_transform():
        return Compose([RandomCrop(224, pad_if_needed=True), ToTensor()])

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ImageTrans Config")
        parser.add_argument('--no_pretrained_resnet', action='store_false')
        parser.add_argument('--z_per_img', type=int, default=5)
        parser.add_argument('--trans_dim', type=int, default=512)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--num_encoder_layers', type=int, default=8)
        parser.add_argument('--num_decoder_layers', type=int, default=8)
        parser.add_argument('--dim_feedforward', type=int, default=2048)
        parser.add_argument('--trans_dropout', type=float, default=0.1)
        parser.add_argument('--layer_norm_eps', type=float, default=1e-6)
        parser.add_argument('--norm_first', action='store_true')
        return parent_parser

    def __init__(self, args, *, out_dim):
        super(ImageTrans, self).__init__()
        self.resnet = nn.Sequential(
            *list(resnet50(pretrained=args.no_pretrained_resnet).children())[
             :-2]
        )
        self.linear = nn.Linear(ImageTrans.seq_dim, args.trans_dim)
        self.transformer = nn.Transformer(
            d_model=args.trans_dim,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.trans_dropout,
            layer_norm_eps=args.layer_norm_eps,
            batch_first=True,
            norm_first=args.norm_first
        )
        self.transformer_param = nn.Parameter(
            torch.empty(args.z_per_img, args.trans_dim))
        self.positional = nn.Parameter(
            torch.empty((ImageTrans.seq_len, args.trans_dim))
        )
        self.linear_sequential = nn.Sequential(
            nn.Linear(args.trans_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )
        self._init_weight()

    def _init_weight(self):
        with torch.no_grad():
            chain = itertools.chain(
                self.transformer_param,
                self.positional,
            )
            for p in chain:
                if p.dim() > 1:
                    xavier_normal_(p)

    def forward(self, x):
        x = self._pass_resnet(x)
        x = self._preprocess_sequence(x)
        x = self._pass_transformer(x)
        x = self.linear_sequential(x)
        return x

    def _pass_transformer(self, x):
        tgt = self.transformer_param
        tgt_e = einops.repeat(tgt, 'lpi model -> b lpi model', b=x.shape[0])
        x = self.transformer(x, tgt_e)
        return x

    def _preprocess_sequence(self, x):
        x = x + self.positional
        return x

    def _pass_resnet(self, x):
        x = self.resnet(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        assert x.shape[1] == ImageTrans.seq_len \
               and x.shape[2] == \
               ImageTrans.seq_dim, 'ResNet feature shape error'
        x = self.linear(x)
        return x
