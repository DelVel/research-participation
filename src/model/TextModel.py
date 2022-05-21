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

import random

import torch
from einops import einops
from torch import nn, Tensor
from torch.nn.init import xavier_normal_
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.transforms import Lambda
from transformers import BertTokenizer


class TextGRU(nn.Module):
    def transform(self, x):
        x = random.sample(x, 5)
        x = self.tokenizer(x, padding='max_length', return_tensors='pt')
        return x['input_ids']

    def get_transform(self):
        return Lambda(self.transform)

    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("TextGRU Config")
        parser.add_argument('--text_embed_dim', type=int, default=256)
        parser.add_argument('--gru_hidden_dim', type=int, default=512)
        parser.add_argument('--gru_num_layers', type=int, default=1)
        parser.add_argument('--gru_dropout', type=float, default=0)
        return parent_parser

    def __init__(self, args, *, out_dim):
        super(TextGRU, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.gru = nn.GRU(
            input_size=args.text_embed_dim,
            hidden_size=args.gru_hidden_dim,
            num_layers=args.gru_num_layers,
            dropout=args.gru_dropout,

            bias=True,
            batch_first=True,
            bidirectional=True
        )
        self.embed = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=args.text_embed_dim,
            padding_idx=self.tokenizer.pad_token_id
        )
        self.linear_sequential = nn.Sequential(
            nn.Linear(2 * args.gru_num_layers * args.gru_hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )

    def _init_weight(self):
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_normal_(p)

    def forward(self, x: Tensor):
        # noinspection PyTypeChecker
        x = self._preprocess_sequence(x)
        x = self._pass_gru(x)
        x = self.linear_sequential(x)
        return x

    def _pass_gru(self, x):
        _, x = self.gru(x)
        x = einops.rearrange(x, 'd_num b model_h -> b (d_num model_h)')
        return x

    def _preprocess_sequence(self, x):
        assert self.tokenizer.pad_token_id == 0, "Padding token should be 0"
        lengths = x.count_nonzero(dim=-1).tolist()
        x = self.embed(x)
        x = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        return x
