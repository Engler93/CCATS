# Copyright 2023 The HuggingFace Team. All rights reserved.
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

# Modifications applied by Philipp Engler for CCATS model
# Original Source: https://github.com/huggingface/diffusers

import math

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.activations import get_activation
from diffusers.models.resnet import Downsample1D, ResidualTemporalBlock1D, Upsample1D, rearrange_dims


class ValueFunctionMidBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.res1 = ResidualTemporalBlock1D(in_channels, in_channels // 2, embed_dim=embed_dim)
        self.down1 = Downsample1D(out_channels // 2, use_conv=True)
        self.res2 = ResidualTemporalBlock1D(in_channels // 2, in_channels // 4, embed_dim=embed_dim)
        self.down2 = Downsample1D(out_channels // 4, use_conv=True)

    def forward(self, x, temb=None):
        x = self.res1(x, temb)
        x = self.down1(x)
        x = self.res2(x, temb)
        x = self.down2(x)
        return x


class OutConv1DBlock(nn.Module):
    def __init__(self, num_groups_out, out_channels, embed_dim, act_fn):
        super().__init__()
        self.final_conv1d_1 = nn.Conv1d(embed_dim, embed_dim, 5, padding=2)
        self.final_conv1d_gn = nn.GroupNorm(num_groups_out, embed_dim)
        self.final_conv1d_act = get_activation(act_fn)
        self.final_conv1d_2 = nn.Conv1d(embed_dim, out_channels, 1)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.final_conv1d_1(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_gn(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_act(hidden_states)
        hidden_states = self.final_conv1d_2(hidden_states)
        return hidden_states


class OutValueFunctionBlock(nn.Module):
    def __init__(self, fc_dim, embed_dim, act_fn="mish"):
        super().__init__()
        self.final_block = nn.ModuleList(
            [
                nn.Linear(fc_dim + embed_dim, fc_dim // 2),
                get_activation(act_fn),
                nn.Linear(fc_dim // 2, 1),
            ]
        )

    def forward(self, hidden_states, temb):
        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        hidden_states = torch.cat((hidden_states, temb), dim=-1)
        for layer in self.final_block:
            hidden_states = layer(hidden_states)

        return hidden_states


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


class Downsample1d(nn.Module):
    def __init__(self, kernel="linear", pad_mode="reflect", learnable=True, num_channels=256):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)
        if learnable:
            self.conv = nn.Conv1d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        else:
            self.conv = None

    def forward(self, hidden_states):
        if self.conv is not None:
            return self.conv(hidden_states)
        hidden_states = F.pad(hidden_states, (self.pad,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros([hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]])
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel
        return F.conv1d(hidden_states, weight, stride=2)


class Upsample1d(nn.Module):
    def __init__(self, kernel="linear", pad_mode="reflect", learnable=True, num_channels=256):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)
        if learnable:
            #self.conv = nn.Conv1d(num_channels, num_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
            self.conv = nn.ConvTranspose1d(num_channels, num_channels, kernel_size=2, stride=2, padding=0, padding_mode='zeros')
        else:
            self.conv = None

    def forward(self, hidden_states, temb=None):
        if self.conv is not None:
            return self.conv(hidden_states)
        hidden_states = F.pad(hidden_states, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros([hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]])
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel
        return F.conv_transpose1d(hidden_states, weight, stride=2, padding=self.pad * 2 + 1)


class SelfAttention1d(nn.Module):
    def __init__(self, in_channels, n_head=1, dropout_rate=0.0, temb_channels=None):
        super().__init__()
        self.channels = in_channels
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)
        self.num_heads = n_head

        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels if temb_channels is None else temb_channels, self.channels)
        self.value = nn.Linear(self.channels if temb_channels is None else temb_channels, self.channels)

        self.proj_attn = nn.Linear(self.channels, self.channels, bias=True)

        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states, temb=None):
        residual = hidden_states
        batch, channel_dim, seq = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        if temb is not None:
            temb = temb.transpose(1, 2)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states if temb is None else temb)
        value_proj = self.value(hidden_states if temb is None else temb)

        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))

        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual
        return output


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, is_last=False, temb_channels=32, cat=False):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = in_channels != out_channels
        self.cat = cat
        if self.has_conv_skip:
            self.conv_skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.temb_proj = nn.Linear(temb_channels, mid_channels)

        self.conv_1 = nn.Conv1d(in_channels, mid_channels, 5, padding=2)
        self.group_norm_1 = nn.GroupNorm(1, mid_channels)
        self.gelu_1 = nn.GELU()
        self.conv_2 = nn.Conv1d(mid_channels if not self.cat else mid_channels*2, out_channels, 5, padding=2)

        if not self.is_last:
            self.group_norm_2 = nn.GroupNorm(1, out_channels)
            self.gelu_2 = nn.GELU()

    def forward(self, hidden_states, temb=None):
        residual = self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        #print(temb)
        #print(temb)
        if temb is not None:
            temb = self.temb_proj(temb.swapaxes(-1,-2)).swapaxes(-1,-2)
            if self.cat:
                hidden_states = torch.cat([hidden_states, temb], dim=1)
            else:
                hidden_states = hidden_states + temb

        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)

        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        output = hidden_states + residual
        return output


class UNetMidBlockAttnCond1D(nn.Module):
    def __init__(self, mid_channels, in_channels, out_channels=None, temb_channels=32, downsampling_fac=1):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        #self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, out_channels, cat=False),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32, temb_channels=temb_channels),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32, temb_channels=temb_channels),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
       # self.up = Upsample1d(kernel="cubic")

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, temb=None):
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        #hidden_states = self.down(hidden_states)
        for i, (attn, resnet) in enumerate(zip(self.attentions, self.resnets)):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states, temb if i in [1,4] else None)

        #hidden_states = self.up(hidden_states)

        return hidden_states


class UNetMidBlock1D(nn.Module):
    def __init__(self, mid_channels, in_channels, out_channels=None, temb_channels=32, downsampling_fac=1):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        #self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, mid_channels, cat=False),
            ResConvBlock(mid_channels, mid_channels, out_channels, cat=False),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
       # self.up = Upsample1d(kernel="cubic")

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, temb=None):
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        #hidden_states = self.down(hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        #hidden_states = self.up(hidden_states)

        return hidden_states


class AttnDownBlock1D(nn.Module):
    def __init__(self, out_channels, in_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic", learnable=False, num_channels=in_channels)
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32, temb_channels=temb_channels),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)

        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states, None if i != 1 else temb)

        return hidden_states, (hidden_states,)


class DownBlock1D(nn.Module):
    def __init__(self, out_channels, in_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("linear", learnable=False, num_channels=in_channels)
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        return hidden_states, (hidden_states,)


class BlockDown1D(nn.Module):
    def __init__(self, out_channels, in_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic", learnable=False, num_channels=out_channels)
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, temb=None):
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//(self.downsampling_fac//2),), mode='nearest')
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        output_states = self.down(hidden_states)

        return output_states, (hidden_states,)


class DownBlock1DNoSkip(nn.Module):
    def __init__(self, out_channels, in_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, temb=None):
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        return hidden_states, (hidden_states,)


class AttnUpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32, temb_channels=temb_channels),
            SelfAttention1d(out_channels, out_channels // 32),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic", learnable=False, num_channels=out_channels)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states, None if i != 1 else temb)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels),
        ]


        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="linear", learnable=False, num_channels=out_channels)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        hidden_states = self.up(hidden_states)

        return hidden_states


class BlockUp1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels),
        ]


        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic", learnable=False, num_channels=out_channels)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        hidden_states = self.up(hidden_states)
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//(self.downsampling_fac//2),), mode='nearest')
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)


        return hidden_states


class UpBlock1DNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels, is_last=True),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        res_hidden_states = res_hidden_states_tuple[-1]
        #print(hidden_states.shape)
        #print(res_hidden_states.shape)
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        return hidden_states


class UpBlock1DReallyNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, temb_channels=None, downsampling_fac=1):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, temb_channels=temb_channels, cat=True),
            ResConvBlock(mid_channels, mid_channels, mid_channels, temb_channels=temb_channels, cat=True),
            ResConvBlock(mid_channels, mid_channels, out_channels, temb_channels=temb_channels, cat=True, is_last=True),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.downsampling_fac = downsampling_fac

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        if temb is not None and temb.shape[-1] > 1:
            temb = F.interpolate(temb, (temb.shape[-1]//self.downsampling_fac,), mode='nearest')
        #res_hidden_states = res_hidden_states_tuple[-1]
        #print(hidden_states.shape)
        #print(res_hidden_states.shape)
        #hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        return hidden_states


def get_down_block(down_block_type, num_layers, in_channels, out_channels, temb_channels, add_downsample, downsampling_fac):
    if down_block_type == "DownBlock1D":
        downsampling_fac = downsampling_fac * 2
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac
    elif down_block_type == "BlockDown1D":
        downsampling_fac = downsampling_fac * 2
        return BlockDown1D(out_channels=out_channels, in_channels=in_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac
    elif down_block_type == "AttnDownBlock1D":
        downsampling_fac = downsampling_fac * 2
        return AttnDownBlock1D(out_channels=out_channels, in_channels=in_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac
    elif down_block_type == "DownBlock1DNoSkip":
        return DownBlock1DNoSkip(out_channels=out_channels, in_channels=in_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(up_block_type, num_layers, in_channels, out_channels, temb_channels, add_upsample, downsampling_fac):
    if up_block_type == "UpBlock1D":
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac//2
    elif up_block_type == "BlockUp1D":
        return BlockUp1D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac//2
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac//2
    elif up_block_type == "UpBlock1DNoSkip":
        return UpBlock1DNoSkip(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac
    elif up_block_type == "UpBlock1DReallyNoSkip":
        return UpBlock1DReallyNoSkip(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, downsampling_fac=downsampling_fac), downsampling_fac
    raise ValueError(f"{up_block_type} does not exist.")


def get_mid_block(mid_block_type, num_layers, in_channels, mid_channels, out_channels, embed_dim, add_downsample, downsampling_fac=1):
    if mid_block_type == "ValueFunctionMidBlock1D":
        return ValueFunctionMidBlock1D(in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim)
    elif mid_block_type == "UNetMidBlock1D":
        return UNetMidBlock1D(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels, temb_channels=embed_dim, downsampling_fac=downsampling_fac)
    elif mid_block_type == "UNetMidBlockAttnCond1D":
        return UNetMidBlockAttnCond1D(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels, temb_channels=embed_dim, downsampling_fac=downsampling_fac)
    raise ValueError(f"{mid_block_type} does not exist.")


def get_out_block(*, out_block_type, num_groups_out, embed_dim, out_channels, act_fn, fc_dim):
    if out_block_type == "OutConv1DBlock":
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    elif out_block_type == "ValueFunction":
        return OutValueFunctionBlock(fc_dim, embed_dim, act_fn)
    return None
