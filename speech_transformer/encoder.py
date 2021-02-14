# Copyright (c) 2021, Sangchun Ha. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
from torch import Tensor
from speech_transformer.attention import MultiHeadAttention
from speech_transformer.mask import (
    MaskConv,
    get_attn_pad_mask,
)
from speech_transformer.module import (
    AddNorm,
    PositionWiseFeedForward,
    PositionWiseFeedForwardConv,
)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
            self,
            num_vocabs: int,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout: float = 0.3,
            max_len: int = 200,
            ff_type: str = 'ff',
            input_size: int = 80,
            use_joint_ctc_attention: bool = False
    ) -> None:
        super(Encoder, self).__init__()
        input_size = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        input_size = int(math.floor(input_size + 2 * 10 - 21) / 2 + 1)
        input_size <<= 5
        self.input_size = input_size
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_encoding = nn.Embedding(max_len, model_dim)
        self.input_fc = nn.Linear(input_size, model_dim)
        self.use_joint_ctc_attention = use_joint_ctc_attention
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True)
        ))
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, ff_dim, num_heads, dropout, ff_type) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(model_dim, num_vocabs)

    def forward(
            self,
            inputs: Tensor,
            inputs_lens: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, list]:
        enc_output_prob = None

        inputs = inputs.unsqueeze(1)

        conv_output, output_lens = self.conv(inputs, inputs_lens)

        conv_output = conv_output.permute(0, 3, 1, 2)
        batch, seq_len, num_channels, hidden_size = conv_output.size()
        conv_output = conv_output.contiguous().view(batch, seq_len, -1)

        self_attn_mask = get_attn_pad_mask(conv_output, output_lens, seq_len)

        position = torch.arange(0, seq_len).unsqueeze(0).repeat(batch, 1)
        inputs = self.layer_norm(self.input_fc(conv_output) + self.positional_encoding(position))
        outputs = self.input_dropout(inputs)

        enc_self_attns = list()
        for encoder_layer in self.encoder_layers:
            outputs, attn_distribution = encoder_layer(outputs, self_attn_mask)
            enc_self_attns.append(attn_distribution)

        if self.use_joint_ctc_attention:
            enc_output_prob = self.fc(outputs)
            enc_output_prob = F.log_softmax(enc_output_prob, dim=-1)  # (B, T, num_vocabs)

        return outputs, output_lens, enc_output_prob, enc_self_attns


class EncoderLayer(nn.Module):
    def __init__(
            self,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_heads: int = 8,
            dropout: float = 0.3,
            ff_type: str = 'ff',
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.add_norm = AddNorm(model_dim)
        if ff_type == 'ff':
            self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim, dropout)
        elif ff_type == 'conv':
            self.feed_forward = PositionWiseFeedForwardConv(model_dim, ff_dim)

    def forward(self, inputs: Tensor, self_attn_mask):
        self_attn_output, attn_distribution = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        self_attn_output = self.add_norm(self_attn_output, inputs)

        ff_output = self.feed_forward(self_attn_output)
        output = self.add_norm(ff_output, self_attn_output)

        return output, attn_distribution

