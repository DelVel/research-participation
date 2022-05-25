#  Copyright 2022 https://github.com/yalesong/pvse
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
import nltk
import numpy as np
import torch
import torchtext
from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor, \
    RandomResizedCrop, RandomHorizontalFlip, Lambda

from src.datamodule import vocab


def bt_to_b1t(target: Tensor):
    return rearrange(target, 'b t -> b 1 t')


class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PIENet(nn.Module):
    """Polysemous Instance Embedding (PIE) module"""

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(PIENet, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.num_embeds > 1:
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.layer_norm(out + residual)
        return out, attn, residual


class PIEImage(nn.Module):
    @classmethod
    def add_module_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--img_num_embeds', type=int, default=1)
        parser.add_argument('--img_dropout', type=float, default=0)
        parser.add_argument('--img_finetune', action='store_true')
        return parent_parser

    def __init__(self, args, *, out_dim):
        super(PIEImage, self).__init__()

        embed_size = out_dim
        self.num_embeds = args.img_num_embeds
        args_dropout = args.img_dropout
        finetune = args.img_finetune

        self.use_attention = True

        # Backbone CNN
        self.cnn = resnet50(pretrained=True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = nn.Linear(cnn_dim, embed_size)
        self.cnn.fc = nn.Sequential()

        self.dropout = nn.Dropout(args_dropout)

        self.pie_net = PIENet(self.num_embeds, cnn_dim, embed_size,
                              cnn_dim // 2, args_dropout)

        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = finetune

        self.stash = None

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        out = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = self.fc(out)
        out = self.dropout(out)

        # compute self-attention map
        out_7x7 = out_7x7.view(-1, self.cnn_dim, 7 * 7)
        out, attn, residual = self.pie_net(out, out_7x7.transpose(1, 2))

        out = F.normalize(out, p=2, dim=-1)
        residual = F.normalize(residual, p=2, dim=-1)
        if self.num_embeds == 1:
            out = bt_to_b1t(out)
            residual = bt_to_b1t(residual)

        self.stash = attn, residual
        return out

    @staticmethod
    def get_transform():
        return Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_collate(img):
        return torch.stack(img)


class PIEText(nn.Module):
    @classmethod
    def add_module_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--txt_num_embeds', type=int, default=1)
        parser.add_argument('--txt_dropout', type=float, default=0)
        parser.add_argument('--txt_finetune', action='store_true')
        return parent_parser

    def __init__(self, args, *, out_dim):
        super(PIEText, self).__init__()

        word_dim = 300
        word2idx = vocab.word2idx
        embed_size = out_dim
        self.num_embeds = args.txt_num_embeds
        args_dropout = args.txt_dropout
        finetune = args.txt_finetune

        self.embed_size = embed_size

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)
        self.embed.weight.requires_grad = finetune

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_size // 2, bidirectional=True,
                          batch_first=True)

        self.pie_net = PIENet(self.num_embeds, word_dim, embed_size,
                              word_dim // 2, args_dropout)
        self.dropout = nn.Dropout(args_dropout)

        self._init_weights(word2idx, word_dim)
        self.stash = None

    def _init_weights(self, word2idx, word_dim):
        # Load pretrained word embedding
        wemb = torchtext.vocab.GloVe()
        assert wemb.vectors.shape[1] == word_dim

        # quick-and-dirty trick to improve word-hit rate
        missing_words = []
        for word, idx in word2idx.items():
            if word not in wemb.stoi:
                word = word \
                    .replace('-', '').replace('.', '').replace("'", '')
                if '/' in word:
                    word = word.split('/')[0]
            if word in wemb.stoi:
                self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
            else:
                missing_words.append(word)
        print(f'Words: {len(word2idx) - len(missing_words)}/{len(word2idx)} '
              f'found in vocabulary; {len(missing_words)} words missing')

    def forward(self, x):
        x, lengths = x
        # Embed word ids to vectors
        wemb_out = self.embed(x)
        wemb_out = self.dropout(wemb_out)

        # Forward propagate RNNs
        lengths = lengths.cpu()
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True,
                                      enforce_sorted=False)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()

        _, rnn_out = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(-1,
                                                             self.embed_size)

        out = self.dropout(rnn_out)

        pad_mask = self._get_pad_mask(wemb_out.shape[1], lengths)
        pad_mask = pad_mask.to(wemb_out.device)
        out, attn, residual = self.pie_net(out, wemb_out, pad_mask)

        out = F.normalize(out, p=2, dim=-1)
        residual = F.normalize(residual, p=2, dim=-1)
        if self.num_embeds == 1:
            out = bt_to_b1t(out)
            residual = bt_to_b1t(residual)

        self.stash = attn, residual
        return out

    def get_transform(self):
        return Lambda(self.transform)

    @staticmethod
    def transform(x):
        sentence = x[0]
        tokens = nltk.tokenize.word_tokenize(str(sentence).lower())
        sentence = [vocab('<start>')]
        sentence.extend([vocab(token) for token in tokens])
        sentence.append(vocab('<end>'))
        target = torch.Tensor(sentence)
        return target

    @staticmethod
    def get_collate(txt):
        cap_lengths = torch.tensor([len(cap) for cap in txt])
        targets = torch.zeros(len(txt), max(cap_lengths)).long()
        for i, cap in enumerate(txt):
            end = cap_lengths[i]
            targets[i, :end] = cap[:end]
        return targets, cap_lengths

    @staticmethod
    def _get_pad_mask(max_length: int, lengths: Tensor):
        ind = torch.arange(0, max_length).unsqueeze(0)
        mask = (ind >= lengths.unsqueeze(1))
        return mask
