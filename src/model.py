import einops
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet50

from src.vocab import padding_idx, vocab_size, padding_len


class ImageTrans(nn.Module):
    def __init__(self, d_model=256, tgt_len=5):
        super(ImageTrans, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            batch_first=True
        )
        self.transformer_param = nn.Parameter(torch.zeros(tgt_len, d_model))
        self.positional = nn.Parameter(
            torch.zeros(ResNetFeature.sequence_length, d_model))

    def forward(self, x):
        x = x + self.positional
        tgt = self.transformer_param
        tgt_e = einops.repeat(tgt, 'l_per_img model -> batch l_per_img model',
                              batch=x.shape[0])
        x = self.transformer(x, tgt_e)
        return x


class ResNetFeature(nn.Module):
    sequence_length = 49
    sequence_dim = 2048

    def __init__(self, pretrained=True):
        super(ResNetFeature, self).__init__()
        self.resnet = nn.Sequential(
            *list(resnet50(pretrained=pretrained).children())[:-2]
        )

    def forward(self, x):
        x = self.resnet(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        assert x.shape[1] == ResNetFeature.sequence_length \
               and x.shape[2] == \
               ResNetFeature.sequence_dim, 'ResNet feature shape error'
        return x


class TextGRU(nn.Module):
    def __init__(self, text_embed_dim=128, hidden_size=128):
        super(TextGRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, text_embed_dim,
                                  padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=text_embed_dim, hidden_size=hidden_size,
                          num_layers=1, bias=True, batch_first=True,
                          dropout=0, bidirectional=True)
        self.positional = nn.Parameter(
            torch.zeros(padding_len, text_embed_dim))

    def forward(self, x: Tensor):
        assert padding_idx == 0, 'count_nonzero assumes padding_idx is 0.'
        lengths = x.count_nonzero(dim=-1).tolist()
        x = self.embed(x) + self.positional
        x = pack_padded_sequence(x, lengths, batch_first=True,
                                 enforce_sorted=False)
        _, x = self.gru(x)
        x = einops.rearrange(x,
                             'hidden batch d_model -> batch (hidden d_model)')
        return x
