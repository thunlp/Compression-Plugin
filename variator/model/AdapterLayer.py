import bmtrain as bmt
import torch
from torch import nn
import torch.nn.functional as F
from model_center.layer import Linear
import math

class LowRankLinear(bmt.DistributedModule):
    #  ------------------------------------------------------------------------------------------
    #  Copyright (c) Microsoft Corporation. All rights reserved.
    #  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
    #  ------------------------------------------------------------------------------------------
    #  copy from loralib and do some refactor
    def __init__(self,
        in_features,
        out_features,
        r=8,
        lora_alpha=16,
        dtype=torch.half,
        activation=False
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        self.activation = activation
        if r > 0:
            self.lora_A = bmt.DistributedParameter(
                torch.empty((r, in_features), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.kaiming_uniform_, a=math.sqrt(5))
            )
            self.lora_B = bmt.DistributedParameter(
                torch.empty((out_features, r), dtype=dtype),
                init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
            )
            if self.activation:
                self.act = nn.ReLU()
            self.scaling = self.lora_alpha / self.r


    def forward(self, x):
        if self.activation:
            return F.linear(self.act(F.linear(x, self.lora_A)), self.lora_B) * self.scaling
        else:
            return F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        # return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class Adapter(torch.nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super().__init__()

        self.w_in = Linear(
            dim_in=dim_in,
            dim_out=dim_mid,
            init_std=0.01,
            bias=True
        )
        self.w_out = Linear(
            dim_in=dim_mid,
            dim_out=dim_out,
            init_std=0.01,
            bias=True
        )
        self.act = torch.nn.Tanh()

    def forward(self, hidden_states: torch.Tensor):
        x = self.w_in(hidden_states)
        x = self.act(x)
        x = self.w_out(x)
        return x
