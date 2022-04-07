#  Copyright 2022 https://github.com/krasserm/perceiver-io
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
import torch.nn as nn
from einops import repeat
from torch.nn import ModuleList

from .adapter import InputAdapter, OutputAdapter
from .utils import Sequential


def mlp(num_channels: int):
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )


def cross_attention_layer(
        num_q_channels: int, num_kv_channels: int, num_heads: int,
        dropout: float
):
    layer = Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads,
                                dropout), dropout),
        Residual(mlp(num_q_channels), dropout),
    )
    return layer


def self_attention_layer(num_channels: int, num_heads: int, dropout: float):
    layer = Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout),
        Residual(mlp(num_channels), dropout)
    )
    return layer


def self_attention_block(
        num_layers: int, num_channels: int, num_heads: int, dropout: float
):
    layers = [self_attention_layer(num_channels, num_heads, dropout) for _ in
              range(num_layers)]
    return Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_q_channels: int, num_kv_channels: int,
                 num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            dropout=dropout,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask,
                              attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):
    # Simplified version of cross-attention module described in
    # https://arxiv.org/abs/2103.03206. Here, the embedding dimension is
    # determined by the number of query channels (num_q_channels) whereas in
    # the paper it can be specified separately. This simplification allows
    # re-use of the torch.nn.MultiHeadAttention module whereas a full
    # implementation of the paper would require a custom multi-head
    # attention implementation.
    def __init__(self, num_q_channels: int, num_kv_channels: int,
                 num_heads: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_q_channels, num_kv_channels=num_kv_channels,
            num_heads=num_heads, dropout=dropout
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask,
                              attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_channels, num_kv_channels=num_channels,
            num_heads=num_heads, dropout=dropout
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class PerceiverEncoder(nn.Module):
    def __init__(
            self,
            input_adapter: InputAdapter,
            num_latent_channels: int,
            num_layers: int = 3,
            num_cross_attention_heads: int = 4,
            num_self_attention_heads: int = 4,
            num_self_attention_layers_per_block: int = 6,
            dropout: float = 0.0,
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific
            input to an encoder input of shape (B, M, C_input) where B is the
            batch size, M the input sequence length and C_input the number of
            input channels.
        :param num_latent_channels: Number of latent channels (C_latent).
        :param num_layers: Number of encoder layers. An encoder layer is
            composed of a cross-attention layer and several self-attention
            layers (= a self-attention block).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_layers_per_block: Number of self-attention
            layers per self-attention block.
        :param dropout: Dropout for self- and cross-attention layers and
            residuals.
        """
        super().__init__()

        self.input_adapter = input_adapter
        self.num_layers = num_layers

        def create_perceiver_layer():
            return Sequential(
                cross_attention_layer(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=input_adapter.num_input_channels,
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                ),
                self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                ),
            )

        self.layers = nn.ModuleList(
            create_perceiver_layer() for _ in range(num_layers))

    def forward(self, latent, x):
        b, *_ = x.shape

        # encode task-specific input
        x, pad_mask = self.input_adapter(x)

        for a_layer in self.layers:
            latent = a_layer(latent, x, pad_mask)

        return latent


class PerceiverEncoderPack(nn.Module):
    def __init__(self, num_latents: int, num_latent_channels: int):
        super().__init__()
        self.num_latent_channels = num_latent_channels
        self.encoders = ModuleList()
        self.latent = nn.Parameter(
            torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def add_encoder(self,
                    input_adapter: InputAdapter,
                    num_layers: int = 3,
                    num_cross_attention_heads: int = 4,
                    num_self_attention_heads: int = 4,
                    num_self_attention_layers_per_block: int = 6,
                    dropout: float = 0.0):
        """Add a PerceiverEncoder to the pack.

        :param input_adapter: Transforms and position-encodes task-specific
            input to an encoder input of shape (B, M, C_input) where B is the
            batch size, M the input sequence length and C_input the number of
            input channels.
        :param num_layers: Number of encoder layers. An encoder layer is
            composed of a cross-attention layer and several self-attention
            layers (= a self-attention block).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_layers_per_block: Number of self-attention
            layers per self-attention block.
        :param dropout: Dropout for self- and cross-attention layers and
            residuals.
        """
        self.encoders.append(
            PerceiverEncoder(
                input_adapter=input_adapter,
                num_latent_channels=self.num_latent_channels,
                num_layers=num_layers,
                num_cross_attention_heads=num_cross_attention_heads,
                num_self_attention_heads=num_self_attention_heads,
                num_self_attention_layers_per_block=
                num_self_attention_layers_per_block,
                dropout=dropout,
            )
        )

    def forward(self, inputs):
        """Forward pass.

        :param inputs: An iterable of x.
        :return: A list of latent vectors of shape (B, num_latents,
            num_latent_channels).
        """
        b, *_ = inputs[0].shape

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        for i, encoder in enumerate(self.encoders):
            x_latent = encoder(x_latent, inputs[i])

        return x_latent


class PerceiverDecoder(nn.Module):
    def __init__(
            self,
            output_adapter: OutputAdapter,
            num_latent_channels: int,
            num_cross_attention_heads: int = 4,
            dropout: float = 0.0,
    ):
        """Generic Perceiver IO decoder.

        :param output_adapter: Transforms generic decoder output of shape
            (B, K, C_output) to task-specific output. B is the batch size,
            K the output sequence length and C_output the number of output
            channels. (K, C_output) is specified via the output_shape property
            of the output_adapter.
        :param num_latent_channels: Number of latent channels (C_latent)
            as produced by a Perceiver IO encoder.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param dropout: Dropout for cross-attention layers and residuals.
        """
        super().__init__()

        num_output_channels = output_adapter.output_shape[-1]

        self.output_adapter = output_adapter
        self.cross_attention = cross_attention_layer(
            num_q_channels=num_output_channels,
            num_kv_channels=num_latent_channels,
            num_heads=num_cross_attention_heads,
            dropout=dropout,
        )

        self.output = nn.Parameter(torch.empty(*output_adapter.output_shape))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *_ = x.shape

        output = repeat(self.output, "... -> b ...", b=b)
        output = self.cross_attention(output, x)
        return self.output_adapter(output)


class PerceiverIO(Sequential):
    def __init__(self, encoder: PerceiverEncoderPack,
                 decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)

    @property
    def encoder_pack(self):
        return self[0]

    @property
    def decoder(self):
        return self[1]
