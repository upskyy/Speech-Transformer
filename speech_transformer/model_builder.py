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

from speech_transformer.encoder import Encoder
from speech_transformer.decoder import Decoder
from speech_transformer.model import SpeechTransformer
import torch


def build_transformer(
        device: torch.device,
        num_vocabs: int,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 5000,
        ff_type: str = 'ff',
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        input_size: int = 80,
        use_joint_ctc_attention: bool = False,
) -> SpeechTransformer:
    encoder = build_encoder(
        num_vocabs,
        model_dim,
        ff_dim,
        num_layers,
        num_heads,
        dropout,
        max_len,
        ff_type,
        input_size,
        use_joint_ctc_attention,
    )
    decoder = build_decoder(
        device,
        num_vocabs,
        model_dim,
        ff_dim,
        num_layers,
        num_heads,
        dropout,
        max_len,
        ff_type,
        pad_id,
        sos_id,
        eos_id,
    )
    return SpeechTransformer(encoder, decoder).to(device)


def build_encoder(
        num_vocabs: int,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 5000,
        ff_type: str = 'ff',
        input_size: int = 80,
        use_joint_ctc_attention: bool = False,
) -> Encoder:
    return Encoder(
        num_vocabs,
        model_dim,
        ff_dim,
        num_layers,
        num_heads,
        dropout,
        max_len,
        ff_type,
        input_size,
        use_joint_ctc_attention,
    )


def build_decoder(
        device: torch.device,
        num_vocabs: int,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 5000,
        ff_type: str = 'ff',
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
) -> Decoder:
    return Decoder(
        device,
        num_vocabs,
        model_dim,
        ff_dim,
        num_layers,
        num_heads,
        dropout,
        max_len,
        ff_type,
        pad_id,
        sos_id,
        eos_id,
    )
