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
from speech_transformer.encoder import Encoder
from speech_transformer.decoder import Decoder
import torch
import torch.nn as nn


class SpeechTransformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(SpeechTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            enc_inputs: Tensor,
            enc_inputs_lens: Tensor,
            dec_inputs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, list, list, list]:
        enc_outputs, enc_output_lens, enc_output_prob, enc_self_attns = self.encoder(enc_inputs, enc_inputs_lens)
        dec_output_prob, dec_self_attns, enc_dec_attns = self.decoder(dec_inputs, enc_outputs, enc_output_lens)

        return enc_output_prob, enc_output_lens, dec_output_prob, enc_self_attns, dec_self_attns, enc_dec_attns

    @torch.no_grad()
    def recognize(self, inputs: Tensor, inputs_lens: Tensor) -> Tensor:
        encoder_outputs, encoder_output_lens, _, _ = self.encoder(inputs, inputs_lens)

        return self.decoder.decode(encoder_outputs, encoder_output_lens)
