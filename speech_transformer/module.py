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
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(
            self,
            model_dim: int = 512,
            ff_dim: int = 2048,
            dropout: float = 0.3,
    ) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.feed_forward(inputs)


class PositionWiseFeedForwardConv(nn.Module):
    def __init__(
            self,
            model_dim: int = 512,
            ff_dim: int = 2048,
    ) -> None:
        super(PositionWiseFeedForwardConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=1, stride=1)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs.transpose(1, 2)  # (B, model_dim, T)
        inputs = self.conv1(inputs)  # (B, ff_dim, T)
        inputs = self.relu(inputs)  # (B, ff_dim, T)
        output = self.conv2(inputs).transpose(1, 2)  # (B, T, model_dim)

        return output


class AddNorm(nn.Module):
    def __init__(self, model_dim: int = 512) -> None:
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, output: Tensor, residual: Tensor) -> Tensor:
        return self.layer_norm(output + residual)

