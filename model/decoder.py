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
from attention import MultiHeadAttention
from mask import (
    get_attn_pad_mask,
    get_decoder_self_attn_mask,
)
from module import (
    AddNorm,
    PositionWiseFeedForward,
    PositionWiseFeedForwardConv,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Decoder(nn.Module):
    def __init__(
            self,
            device: torch.device,
            num_vocabs: int,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout: float = 0.3,
            max_len: int = 200,
            ff_type: str = 'ff',
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(num_vocabs, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = nn.Embedding(max_len, model_dim)
        self.input_dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, ff_dim, num_heads, dropout, ff_type) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(model_dim, num_vocabs)

    def forward(
            self,
            inputs: Tensor,
            encoder_output: Tensor,
            encoder_output_lens: Tensor
    ) -> Tuple[Tensor, list, list]:
        batch = inputs.size(0)

        if len(inputs.size()) == 1:  # validate, evaluation
            inputs = inputs.unsqueeze(1)
        else:  # train
            inputs = inputs[inputs != self.eos_id].view(batch, -1)

        target_lens = inputs.size(1)
        position = torch.arange(0, target_lens).unsqueeze(0).repeat(batch, 1).to(self.device)

        enc_dec_attn_mask = get_attn_pad_mask(encoder_output, encoder_output_lens, target_lens)
        self_attn_mask = get_decoder_self_attn_mask(inputs, self.pad_id)

        embedding_output = self.embedding(inputs).to(self.device) * self.scale
        positional_encoding_output = self.positional_encoding(position)

        inputs = embedding_output + positional_encoding_output
        outputs = self.input_dropout(inputs)

        dec_self_attns = list()
        enc_dec_attns = list()
        for decoder_layer in self.decoder_layers:
            outputs, self_attn_distribution, enc_dec_attn_distribution = decoder_layer(outputs,
                                                                                       encoder_output,
                                                                                       self_attn_mask,
                                                                                       enc_dec_attn_mask)
            dec_self_attns.append(self_attn_distribution)
            enc_dec_attns.append(enc_dec_attn_distribution)

        decoder_output = self.fc(outputs)
        decoder_output_prob = F.log_softmax(decoder_output, dim=-1)

        return decoder_output_prob, dec_self_attns, enc_dec_attns

    @torch.no_grad()
    def decode(self, encoder_outputs: Tensor, encoder_output_lens: Tensor) -> Tensor:
        batch = encoder_outputs.size(0)
        y_hat = list()

        inputs = torch.LongTensor([self.sos_id] * batch)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        for _ in range(0, self.max_len):
            dec_output_prob, _, _ = self.forward(inputs, encoder_outputs, encoder_output_lens)
            dec_output_prob = dec_output_prob.squeeze(1)
            inputs = dec_output_prob.max(1)[1]
            y_hat.append(inputs)

        y_hat = torch.stack(y_hat, dim=1)

        return y_hat  # (B, T)


class DecoderLayer(nn.Module):
    def __init__(
            self,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_heads: int = 8,
            dropout: float = 0.3,
            ff_type: str = 'ff',
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.encoder_decoder_attention = MultiHeadAttention(model_dim, num_heads)
        self.add_norm = AddNorm(model_dim)
        if ff_type == 'ff':
            self.feed_forward = PositionWiseFeedForward(model_dim, ff_dim, dropout)
        elif ff_type == 'conv':
            self.feed_forward = PositionWiseFeedForwardConv(model_dim, ff_dim)

    def forward(
            self,
            inputs: Tensor,
            encoder_output: Tensor,
            self_attn_mask: Tensor,
            enc_dec_attn_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        self_attn_output, self_attn_distribution = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        self_attn_output = self.add_norm(self_attn_output, inputs)

        enc_dec_attn_output, enc_dec_attn_distribution = self.encoder_decoder_attention(self_attn_output,
                                                                                        encoder_output,
                                                                                        encoder_output,
                                                                                        enc_dec_attn_mask)
        enc_dec_attn_output = self.add_norm(enc_dec_attn_output, self_attn_output)

        ff_output = self.feed_forward(enc_dec_attn_output)
        output = self.add_norm(ff_output, enc_dec_attn_output)

        return output, self_attn_distribution, enc_dec_attn_distribution
