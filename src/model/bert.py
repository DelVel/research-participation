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

from torch import nn
from torchvision.transforms import Lambda
from transformers import DistilBertTokenizer, DistilBertModel


class BERT(nn.Module):
    def transform(self, x):
        x = x[0]
        x = self.tokenizer(x, padding='max_length', return_tensors='pt')
        for k in x:
            x[k].squeeze_(0)
        return x

    def get_transform(self):
        return Lambda(self.transform)

    @staticmethod
    def add_module_specific_args(parent_parser):
        return parent_parser

    def __init__(self, arg, *, out_dim):
        super().__init__()
        self.tokenizer = DistilBertTokenizer \
            .from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.linear = nn.Linear(768, out_dim)

    def forward(self, x):
        x = self.model(**x)
        x = self.linear(x[0][:, :1, :])
        return x
