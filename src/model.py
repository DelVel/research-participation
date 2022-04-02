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

import einops
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet50

from src.vocab import padding_idx, vocab_size


class ImageTrans(nn.Module):
    seq_len = 49
    seq_dim = 2048

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

    def __init__(self, *,
                 out_dim,

                 pretrained,
                 zpi,
                 trans_dim,

                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 dropout,
                 layer_norm_eps,
                 norm_first
                 ):
        super(ImageTrans, self).__init__()
        self.resnet = nn.Sequential(
            *list(resnet50(pretrained=pretrained).children())[:-2]
        )
        self.linear = nn.Linear(ImageTrans.seq_dim, trans_dim)
        self.transformer = nn.Transformer(
            d_model=trans_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer_param = nn.Parameter(torch.zeros(zpi, trans_dim))
        self.positional = nn.Parameter(
            torch.zeros((ImageTrans.seq_len, trans_dim))
        )
        self.linear_sequential = nn.Sequential(
            nn.Linear(trans_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )

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


class TextGRU(nn.Module):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TextGRU Config")
        parser.add_argument('--text_embed_dim', type=int, default=256)
        parser.add_argument('--gru_hidden_dim', type=int, default=512)
        parser.add_argument('--gru_num_layers', type=int, default=1)
        parser.add_argument('--gru_dropout', type=float, default=0)
        return parent_parser

    def __init__(self, *,
                 out_dim,

                 text_embed_dim,
                 gru_hidden_size,
                 gru_layers,
                 dropout,
                 ):
        super(TextGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=text_embed_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            dropout=dropout,

            bias=True,
            batch_first=True,
            bidirectional=True
        )
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=text_embed_dim,
            padding_idx=padding_idx
        )
        self.linear_sequential = nn.Sequential(
            nn.Linear(2 * gru_layers * gru_hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x: Tensor):
        x = self._preprocess_sequence(x)
        x = self._pass_gru(x)
        x = self.linear_sequential(x)
        return x

    def _pass_gru(self, x):
        _, x = self.gru(x)
        x = einops.rearrange(x, 'd_num b model_h -> b (d_num model_h)')
        return x

    def _preprocess_sequence(self, x):
        assert padding_idx == 0, 'count_nonzero assumes padding_idx is 0.'
        lengths = x.count_nonzero(dim=-1).tolist()
        x = self.embed(x)
        x = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        return x
