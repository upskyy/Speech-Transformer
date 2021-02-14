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

from torch import Tensor
from typing import Tuple
import torch
import torch.nn as nn


class MaskConv(nn.Module):
    """
    Masking Convolutional Neural Network

    Refer to https://github.com/sooftware/KoSpeech/blob/jasper/kospeech/models/modules.py
    Copyright (c) 2020 Soohwan Kim
    """

    def __init__(
            self,
            sequential: nn.Sequential,
    ) -> None:
        super(MaskConv, self).__init__()
        self.sequential = sequential

    def forward(
            self,
            inputs: Tensor,
            seq_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        output = None

        for module in self.sequential:
            output = module(inputs)

            mask = torch.BoolTensor(output.size()).fill_(0)
            if output.is_cuda:
                mask = mask.cuda()

            seq_lens = self.get_seq_lens(module, seq_lens)

            for idx, seq_len in enumerate(seq_lens):
                seq_len = seq_len.item()

                if (mask[idx].size(2) - seq_len) > 0:
                    mask[idx].narrow(2, seq_len, mask[idx].size(2) - seq_len).fill_(1)

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lens

    def get_seq_lens(
            self,
            module: nn.Module,
            seq_lens: Tensor,
    ) -> Tensor:
        if isinstance(module, nn.Conv2d):
            seq_lens = seq_lens + (2 * module.padding[1]) - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lens = seq_lens.float() / float(module.stride[1])
            seq_lens = seq_lens.int() + 1

        if isinstance(module, nn.MaxPool2d):
            seq_lens >>= 1

        return seq_lens.int()


def _get_pad_mask(inputs: Tensor, inputs_lens: Tensor):
    assert len(inputs.size()) == 3

    batch = inputs.size(0)

    pad_attn_mask = inputs.new_zeros(inputs.size()[: -1])

    for idx in range(batch):
        pad_attn_mask[idx, inputs_lens[idx]:] = 1

    return pad_attn_mask.bool()


def get_attn_pad_mask(inputs: Tensor, inputs_lens: Tensor, expand_lens):
    pad_attn_mask = _get_pad_mask(inputs, inputs_lens)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, expand_lens, 1)  # (batch, dec_T, enc_T)

    return pad_attn_mask


def _get_attn_key_pad_mask(target: Tensor, pad_id: int):
    target_lens = target.size(1)
    padding_mask = target.eq(pad_id)
    padding_mask = padding_mask.unsqueeze(1).repeat(1, target_lens, 1)

    return padding_mask


def _get_subsequent_mask(target: Tensor):
    batch, target_lens = target.size()
    subsequent_mask = torch.triu(torch.ones((target_lens, target_lens), device=target.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch, 1, 1)

    return subsequent_mask


def get_decoder_self_attn_mask(target: Tensor, pad_id: int = 0):
    padding_mask = _get_attn_key_pad_mask(target, pad_id)
    subsequent_mask = _get_subsequent_mask(target)

    decoder_self_attn_mask = (padding_mask + subsequent_mask).bool()

    return decoder_self_attn_mask
