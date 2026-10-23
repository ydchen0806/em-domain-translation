# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from monai.utils import ensure_tuple_rep
from monai.networks.blocks import Convolution
from monai.networks.nets.controlnet import ControlNet
from monai.networks.nets.diffusion_model_unet import get_timestep_embedding

from networks_copy.diffusion_model_unet import get_down_block

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ControlNetConditioningEmbedding(nn.Module):
    """
    Network to encode the conditioning into a latent space.
    """

    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, channels: Sequence[int]):
        super().__init__()

        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            adn_ordering="A",
            act="SWISH",
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(channels) - 1): # 3
            channel_in = channels[i]
            channel_out = channels[i + 1]
            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_in,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    adn_ordering="A",
                    act="SWISH",
                )
            )

            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_out,
                    strides=[2,2,1],
                    kernel_size=3,
                    padding=1,
                    adn_ordering="A",
                    act="SWISH",
                )
            )

        self.conv_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

    def forward(self, conditioning):
        # conditioning torch.Size([1, 3, 160, 160, 16])
        embedding = self.conv_in(conditioning)
        # embedding torch.Size([1, 8, 160, 160, 16])

        for block in self.blocks:
            embedding = block(embedding)
            '''
            embedding block torch.Size([1, 8, 160, 160, 16])
            embedding block torch.Size([1, 32, 80, 80, 8])
            embedding block torch.Size([1, 32, 80, 80, 8])
            embedding block torch.Size([1, 64, 40, 40, 4])
            '''

        embedding = self.conv_out(embedding)
        # embedding out torch.Size([1, 64, 40, 40, 4])

        return embedding
    
class ControlNetMaisi(ControlNet):
    """
    Control network for diffusion models based on Zhang and Agrawala "Adding Conditional Control to Text-to-Image
    Diffusion Models" (https://arxiv.org/abs/2302.05543)

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        num_res_blocks: number of residual blocks (see ResnetBlock) per level.
        num_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds`
            classes.
        upcast_attention: if True, upcast attention operations to full precision.
        conditioning_embedding_in_channels: number of input channels for the conditioning embedding.
        conditioning_embedding_num_channels: number of channels for the blocks in the conditioning embedding.
        use_checkpointing: if True, use activation checkpointing to save memory.
        include_fc: whether to include the final linear layer. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int =3,
        in_channels: int=1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        conditioning_embedding_in_channels: int = 1,
        conditioning_embedding_num_channels: Sequence[int] = (16, 32, 96, 256),
        use_checkpointing: bool = True,
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims,
            in_channels,
            num_res_blocks,
            num_channels,
            attention_levels,
            norm_num_groups,
            norm_eps,
            resblock_updown,
            num_head_channels,
            with_conditioning,
            transformer_num_layers,
            cross_attention_dim,
            num_class_embeds,
            upcast_attention,
            conditioning_embedding_in_channels,
            conditioning_embedding_num_channels,
            include_fc,
            use_combined_linear,
            use_flash_attention,
        )
        self.use_checkpointing = use_checkpointing
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            spatial_dims=spatial_dims, # 3
            in_channels=conditioning_embedding_in_channels, # 3
            channels=conditioning_embedding_num_channels, # [8, 32, 64]
            out_channels=num_channels[0], # 64
        )
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        time_embed_dim = num_channels[0] * 4
        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = int(i) == len(num_channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[int(i)],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
            )

            self.down_blocks.append(down_block)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        emb = self._prepare_time_and_class_embedding(x, timesteps, class_labels)
        h = self._apply_initial_convolution(x)
        if self.use_checkpointing:
            controlnet_cond = torch.utils.checkpoint.checkpoint(
                self.controlnet_cond_embedding, controlnet_cond, use_reentrant=False
            )
        else:
            controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        # h: [b, 64, 40, 40, 16]
        # controlnet_cond: [b, 64, 40, 40, 4] 由于MAISI【这里把stride从2改成1了】
        h = h + controlnet_cond
        down_block_res_samples, h = self._apply_down_blocks(emb, context, h)
        # for i in down_block_res_samples:
        #     print('1 down_block_res_samples', i.shape)
        h = self._apply_mid_block(emb, context, h)
        down_block_res_samples, mid_block_res_sample = self._apply_controlnet_blocks(h, down_block_res_samples)
        # for i in down_block_res_samples:
        #     print('2 down_block_res_samples', i.shape)
        # scaling
        down_block_res_samples = [h * conditioning_scale for h in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        return down_block_res_samples, mid_block_res_sample

    def _prepare_time_and_class_embedding(self, x, timesteps, class_labels):
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb       = emb + class_emb

        return emb

    def _apply_initial_convolution(self, x):
        # 3. initial convolution
        h = self.conv_in(x)
        return h

    def _apply_down_blocks(self, emb, context, h):
        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        return down_block_res_samples, h

    def _apply_mid_block(self, emb, context, h):
        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)
        return h

    def _apply_controlnet_blocks(self, h, down_block_res_samples):
        # 6. Control net blocks
        controlnet_down_block_res_samples = []
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples.append(down_block_res_sample)

        mid_block_res_sample = self.controlnet_mid_block(h)

        return controlnet_down_block_res_samples, mid_block_res_sample
