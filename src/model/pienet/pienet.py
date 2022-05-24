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

import torch
import numpy as np
from torch import nn


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


class EncoderImage(nn.Module):

    def __init__(self, opt):
        super(EncoderImage, self).__init__()

        embed_size, num_embeds = opt.embed_size, opt.num_embeds
        self.use_attention = opt.img_attention
        self.abs = True if hasattr(opt, 'order') and opt.order else False

        # Backbone CNN
        self.cnn = get_cnn(opt.cnn_type, True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = nn.Linear(cnn_dim, embed_size)
        self.cnn.fc = nn.Sequential()

        self.dropout = nn.Dropout(opt.dropout)

        if self.use_attention:
            self.pie_net = PIENet(num_embeds, cnn_dim, embed_size,
                                  cnn_dim // 2, opt.dropout)

        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = opt.img_finetune

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        out = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = self.fc(out)
        out = self.dropout(out)

        # compute self-attention map
        attn, residual = None, None
        if self.use_attention:
            out_7x7 = out_7x7.view(-1, self.cnn_dim, 7 * 7)
            out, attn, residual = self.pie_net(out, out_7x7.transpose(1, 2))

        out = l2norm(out)
        if self.abs:
            out = torch.abs(out)

        return out, attn, residual


class EncoderText(nn.Module):

    def __init__(self, word2idx, opt):
        super(EncoderText, self).__init__()

        wemb_type, word_dim, embed_size, num_embeds = \
            opt.wemb_type, opt.word_dim, opt.embed_size, opt.num_embeds

        self.embed_size = embed_size
        self.use_attention = opt.txt_attention
        self.abs = True if hasattr(opt, 'order') and opt.order else False
        self.legacy = opt.legacy

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)
        self.embed.weight.requires_grad = opt.txt_finetune

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_size // 2, bidirectional=True,
                          batch_first=True)
        if self.use_attention:
            self.pie_net = PIENet(num_embeds, word_dim, embed_size,
                                  word_dim // 2, opt.dropout)
        self.dropout = nn.Dropout(opt.dropout)

        self.init_weights(wemb_type, word2idx, word_dim)

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe()
            else:
                raise Exception(
                    'Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'",
                                                                          '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx),
                len(missing_words)))

    def forward(self, x, lengths):
        # Embed word ids to vectors
        wemb_out = self.embed(x)
        wemb_out = self.dropout(wemb_out)

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()

        # Use legacy mode to reproduce results in CVPR 2018 paper
        if self.legacy:
            rnn_out, _ = self.rnn(packed)
            padded = pad_packed_sequence(rnn_out, batch_first=True)
            I = lengths.expand(self.embed_size, 1, -1).permute(2, 1, 0) - 1
            rnn_out = torch.gather(padded[0], 1, I).squeeze(1)
        else:
            _, rnn_out = self.rnn(packed)
            # Reshape *final* output to (batch_size, hidden_size)
            rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(-1,
                                                                 self.embed_size)

        out = self.dropout(rnn_out)

        attn, residual = None, None
        if self.use_attention:
            pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
            out, attn, residual = self.pie_net(out, wemb_out, pad_mask)

        out = l2norm(out)
        if self.abs:
            out = torch.abs(out)
        return out, attn, residual
