import einops
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet50

from src.vocab import padding_idx, vocab_size, padding_len


class ImageTrans(nn.Module):
    def __init__(self, d_model=256, img_text_granularity=5):
        super(ImageTrans, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True)
        self.transformer_param = nn.Parameter(torch.zeros(img_text_granularity, d_model))
        self.positional = nn.Parameter(torch.zeros(49, d_model))

    def forward(self, x):
        x = x + self.positional
        param_expand = einops.repeat(self.transformer_param, 'l_per_img model -> b l_per_img model', b=x.shape[0])
        x = self.transformer(x, param_expand)
        return x


class ResNet50Feature(nn.Module):
    def __init__(self):
        super(ResNet50Feature, self).__init__()
        self.resnet = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])

    def forward(self, x):
        x = self.resnet(x)
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        return x


class TextGRU(nn.Module):
    def __init__(self, text_embed_dim=128, hidden_size=128):
        super(TextGRU, self).__init__()
        self.embed = nn.Embedding(vocab_size, text_embed_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=text_embed_dim, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True,
                          dropout=0, bidirectional=True)
        self.positional = nn.Parameter(torch.zeros(padding_len, text_embed_dim))

    def forward(self, x):
        lengths = ((x == 2).nonzero()[:, 1] + 1).tolist()
        x = self.embed(x)
        x = x + self.positional
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, x = self.gru(x)
        x = einops.rearrange(x, 'h b l -> b (h l)')
        return x
