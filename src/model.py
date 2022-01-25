import einops
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet50

from src.vocab import padding_idx, vocab_size, padding_len


class ImageTrans(nn.Module):
    seq_len = 49
    seq_dim = 2048

    def __init__(self, *,
                 out_dim=256,

                 pretrained=True,
                 zpi=5,
                 trans_dim=512,

                 nhead=8,
                 num_encoder_layers=8,
                 num_decoder_layers=8,
                 dim_feedforward=2048,
                 dropout=0.1,
                 layer_norm_eps=1e-6,
                 norm_first=False
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
        self.token = nn.Parameter(torch.zeros((trans_dim,)))
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
        expanded = einops.repeat(self.token, 'd -> b 1 d', b=x.shape[0])
        x = torch.cat((x, expanded), dim=1)
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
    def __init__(self, *,
                 out_dim=128,

                 text_embed_dim=128,
                 gru_hidden_size=128,
                 gru_layers=1,
                 bias=True,
                 dropout=0,
                 bidirectional=True
                 ):
        super(TextGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=text_embed_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=text_embed_dim,
            padding_idx=padding_idx
        )
        self.positional = nn.Parameter(
            torch.zeros(padding_len, text_embed_dim)
        )
        coefficient = 2 if bidirectional else 1
        self.linear_sequential = nn.Sequential(
            nn.Linear(coefficient * gru_layers * gru_hidden_size, 2048),
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
        x = self.embed(x) + self.positional
        x = pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        return x
