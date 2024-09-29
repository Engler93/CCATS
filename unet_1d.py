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

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps, get_1d_sincos_pos_embed_from_grid
from diffusers.models.modeling_utils import ModelMixin
from unet_1d_blocks import get_down_block, get_mid_block, get_out_block, get_up_block


@dataclass
class UNet1DOutput(BaseOutput):
    """
    The output of [`UNet1DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, sample_size)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class UNet1DModel(ModelMixin, ConfigMixin):
    r"""
    A 1D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int`, *optional*): Default length of sample. Should be adaptable at runtime.
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        extra_in_channels (`int`, *optional*, defaults to 0):
            Number of additional channels to be added to the input of the first down block. Useful for cases where the
            input data has more channels than what the model was initially designed for.
        time_embedding_type (`str`, *optional*, defaults to `"fourier"`): Type of time embedding to use.
        freq_shift (`float`, *optional*, defaults to 0.0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D")`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(32, 32, 64)`):
            Tuple of block output channels.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock1D"`): Block type for middle of UNet.
        out_block_type (`str`, *optional*, defaults to `None`): Optional output processing block of UNet.
        act_fn (`str`, *optional*, defaults to `None`): Optional activation function in UNet blocks. (output and embedding)
        norm_num_groups (`int`, *optional*, defaults to 8): The number of groups for normalization.
        layers_per_block (`int`, *optional*, defaults to 1): The number of layers per block.
        downsample_each_block (`int`, *optional*, defaults to `False`):
            Experimental feature for using a UNet without upsampling.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 65536,
        sample_rate: Optional[int] = None,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 0,
        time_embedding_type: str = "fourier",
        ts_step_embedding_type: str = None,
        condition_dim: int = 0,
        condition_embedding: str = "embedding",
        flip_sin_to_cos: bool = True,
        encode_embeddings: bool = False,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str] = ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        mid_block_type: Tuple[str] = "UNetMidBlock1D",
        out_block_type: str = None,
        block_out_channels: Tuple[int] = (32, 32, 64),
        act_fn: str = None,
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        downsample_each_block: bool = False,
    ):
        super().__init__()

        max_downsampling_fac = 1
        for block in down_block_types:
            if block == "DownBlock1DNoSkip":
                max_downsampling_fac *= 1
            elif block == "DownBlock1D" or block == "AttnDownBlock1D" or block == "BlockDown1D":
                max_downsampling_fac *= 2
            else:
                raise NotImplementedError("Block type not unknown for in downsampling factor")

        self.sample_size = sample_size
        # sample is padded to ensure that after upsampling has the same length, padding removed at the end again
        self.padded_size = math.ceil(float(sample_size)/max_downsampling_fac)*max_downsampling_fac
        self.start_valid = math.floor(float(self.padded_size - self.sample_size)/2.0)
        self.pad_left = self.start_valid
        self.end_valid = self.start_valid + self.sample_size
        self.pad_right = self.padded_size - self.end_valid

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=8, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = block_out_channels[0]
        else:
            timestep_input_dim = 0
            self.time_proj = None

        # create embedding for time steps of the input time series
        if ts_step_embedding_type == 'positional':
            ts_step_embedding_dim = block_out_channels[0]
            self.ts_step_embedding = torch.from_numpy(get_1d_sincos_pos_embed_from_grid(ts_step_embedding_dim, torch.arange(start=0, end=self.sample_size))).float().swapaxes(-1,-2)
            self.ts_step_embedding = F.pad(self.ts_step_embedding, (self.pad_left, self.pad_right), mode="constant", value=0).unsqueeze(0)
        else:
            ts_step_embedding_dim = 0
            self.ts_step_embedding = None

        if condition_dim is not None and condition_dim>0:
            self.cond_embedding_dim = block_out_channels[0]
            if condition_embedding == "embedding":
                self.cond_embedding = nn.Embedding(condition_dim, self.cond_embedding_dim)
            elif condition_embedding == "linear":
                self.cond_embedding = nn.Linear(condition_dim, self.cond_embedding_dim)
            else:
                raise NotImplementedError
        else:
            self.cond_embedding_dim = 0
            self.cond_embedding = None

        if encode_embeddings:
            first_proj_dim = timestep_input_dim + self.cond_embedding_dim
            assert first_proj_dim > 0
            self.emb_proj1 = nn.Sequential(
                nn.Linear(first_proj_dim, block_out_channels[0] * 2),
                nn.ReLU(),
                nn.Linear(block_out_channels[0] * 2, block_out_channels[0] * 2),
            )
            if ts_step_embedding_dim > 0:
                self.emb_proj2 = nn.Sequential(nn.Linear(block_out_channels[0] * 3, block_out_channels[0] * 2), nn.ReLU())
            else:
                self.emb_proj2 = None
            self.emb_channels = block_out_channels[0] * 2
        else:
            self.emb_proj1 = None
            self.emb_proj2 = None
            self.emb_channels = block_out_channels[0]

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None

        downsampling_fac = 1
        # down
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if i == 0:
                input_channel += extra_in_channels

            is_final_block = i == len(block_out_channels) - 1

            down_block, downsampling_fac = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=self.emb_channels,
                add_downsample=not is_final_block or downsample_each_block,
                downsampling_fac=downsampling_fac
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_dim=self.emb_channels,
            num_layers=layers_per_block,
            add_downsample=downsample_each_block,
            downsampling_fac=downsampling_fac,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        if out_block_type is None:
            final_upsample_channels = out_channels
        else:
            final_upsample_channels = block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (
                reversed_block_out_channels[i + 1] if i < len(up_block_types) - 1 else final_upsample_channels
            )

            is_final_block = i == len(block_out_channels) - 1

            up_block, downsampling_fac = get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=self.emb_channels,
                add_upsample=not is_final_block,
                downsampling_fac=downsampling_fac,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=self.emb_channels,
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )


    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        label: torch.LongTensor = None,
        drop_labels: float = 0.0,
        return_dict: bool = True,
    ) -> Union[UNet1DOutput, Tuple]:
        r"""
        The [`UNet1DModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_1d.UNet1DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_1d.UNet1DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_1d.UNet1DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. padding
        sample = F.pad(sample, (self.pad_left, self.pad_right), mode="replicate")

        # 1. embeddings
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        embedding = None if self.time_proj is None and self.cond_embedding is None and self.ts_step_embedding is None else []
        if self.time_proj is not None:
            temb = self.time_proj(timesteps)
            if temb.shape[0] < sample.shape[0]:
                temb = temb.repeat(sample.shape[0], 1)
            embedding.append(temb)

        if self.cond_embedding is not None:
            if label is not None:
                label_emb = self.cond_embedding(label)
                if drop_labels > 0:
                    mask = (torch.rand(size=(label.shape[0],1)) > drop_labels).int().to(sample.device)
                    label_emb = label_emb * mask
                embedding.append(label_emb)
            else:
                embedding.append(torch.zeros((sample.shape[0], self.cond_embedding_dim), dtype=sample.dtype, device=sample.device))
        elif label is not None:
            raise ValueError("Label given but model has no embedding layer for label")


        if self.emb_proj1:
            # encode the embeddings
            embedding = self.emb_proj1(torch.cat(embedding, dim=1)).unsqueeze(-1)
            if self.emb_proj2:
                ts_step_embedding = self.ts_step_embedding.to(sample.device).broadcast_to([sample.shape[0]] + list(self.ts_step_embedding.shape[1:]))
                embedding = embedding.broadcast_to([sample.shape[0]] + [embedding.shape[-2]] + [ts_step_embedding.shape[-1]])
                # need to swap axes frequently, as linear layers should project channel dimension, which is not last for conv layer representations
                embedding = self.emb_proj2(torch.cat([embedding, ts_step_embedding], dim=1).swapaxes(-1,-2)).swapaxes(-1,-2)

        elif embedding is not None:
            # sum the embeddings
            time_dim = 1 if self.ts_step_embedding is None else self.padded_size
            for i in range(len(embedding)):
                embedding[i] = embedding[i].unsqueeze(-1).broadcast_to(list(embedding[i].shape) + [time_dim])
                #print(i, embedding[i].shape)
            if self.ts_step_embedding is not None:
                ts_step_embedding = self.ts_step_embedding.to(sample.device).broadcast_to([sample.shape[0]] + list(self.ts_step_embedding.shape[1:]))
                embedding.append(ts_step_embedding)
            embedding = torch.stack(embedding, dim=0).sum(dim=0)


        # 2. down
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=embedding)#timestep_embed)
            down_block_res_samples += res_samples
        # 3. mid
        if self.mid_block:
            sample = self.mid_block(sample, temb=None)#timestep_embed)
        mid_output = sample

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=embedding)#timestep_embed)

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, embedding)#timestep_embed)

        sample = sample[:,:, self.start_valid:self.end_valid]

        if not return_dict:
            return (sample,)

        return dict(sample=sample, mid_output=mid_output)
