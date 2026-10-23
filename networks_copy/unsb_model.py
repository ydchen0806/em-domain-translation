from __future__ import annotations

from collections.abc import Sequence

import torch
import numpy as np
from torch import nn


from monai.networks.blocks import Convolution,ADN
# from monai.networks.nets.diffusion_model_unet import (
#     get_down_block,
#     get_mid_block,
#     get_timestep_embedding,
#     get_up_block,
#     zero_module,
# )
from monai.utils import ensure_tuple_rep
from monai.utils.type_conversion import convert_to_tensor


from monai.networks.layers import Act
from monai.networks.utils import normal_init

from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv

from networks_copy.diffusion_model_unet import (
    DiffusionUnetDownsample,
    get_down_block,
    get_mid_block,
    get_timestep_embedding,
    get_up_block,
    zero_module,
)

# __all__ = ["DiffusionModelUNetMaisi"]

use144=True

# 其实用的就是diffuison model
class Generator(nn.Module):

    """
    U-Net network with timestep embedding and attention mechanisms for conditioning based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: Number of spatial dimensions.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_res_blocks: Number of residual blocks (see ResnetBlock) per level. Can be a single integer or a sequence of integers.
        num_channels: Tuple of block output channels.
        attention_levels: List of levels to add attention.
        norm_num_groups: Number of groups for the normalization.
        norm_eps: Epsilon for the normalization.
        resblock_updown: If True, use residual blocks for up/downsampling.
        num_head_channels: Number of channels in each attention head. Can be a single integer or a sequence of integers.
        with_conditioning: If True, add spatial transformers to perform conditioning.
        transformer_num_layers: Number of layers of Transformer blocks to use.
        cross_attention_dim: Number of context dimensions to use.
        num_class_embeds: If specified (as an int), then this model will be class-conditional with `num_class_embeds` classes.
        upcast_attention: If True, upcast attention operations to full precision.
        include_fc: whether to include the final linear layer. Default to False.
        use_combined_linear: whether to use a single linear layer for qkv projection, default to False.
        use_flash_attention: If True, use flash attention for a memory efficient attention mechanism.
        dropout_cattn: If different from zero, this will be the dropout value for the cross-attention layers.
        include_top_region_index_input: If True, use top region index input.
        include_bottom_region_index_input: If True, use bottom region index input.
        include_spacing_input: If True, use spacing input.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
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
        include_fc: bool = False,
        use_combined_linear: bool = False,
        use_flash_attention: bool = False,
        dropout_cattn: float = 0.0,
        include_top_region_index_input: bool = False,
        include_bottom_region_index_input: bool = False,
        include_spacing_input: bool = False,
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelUNetMaisi expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelUNetMaisi expects with_conditioning=True when specifying the cross_attention_dim."
            )
        if dropout_cattn > 1.0 or dropout_cattn < 0.0:
            raise ValueError("Dropout cannot be negative or >1.0!")

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError(
                f"DiffusionModelUNetMaisi expects all num_channels being multiple of norm_num_groups, "
                f"but get num_channels: {num_channels} and norm_num_groups: {norm_num_groups}"
            )

        if len(num_channels) != len(attention_levels):
            raise ValueError(
                f"DiffusionModelUNetMaisi expects num_channels being same size of attention_levels, "
                f"but get num_channels: {len(num_channels)} and attention_levels: {len(attention_levels)}"
            )

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = num_channels[0] * 4 # 64*4=256 # 【注意其实可以修改，比如用1024】
        self.time_embed = self._create_embedding_module(num_channels[0], time_embed_dim)

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        self.include_top_region_index_input = include_top_region_index_input
        self.include_bottom_region_index_input = include_bottom_region_index_input
        self.include_spacing_input = include_spacing_input
        

        new_time_embed_dim = time_embed_dim
        if self.include_top_region_index_input:
            self.top_region_index_layer = self._create_embedding_module(4, time_embed_dim)
            new_time_embed_dim += time_embed_dim # 512
        if self.include_bottom_region_index_input:
            self.bottom_region_index_layer = self._create_embedding_module(4, time_embed_dim)
            new_time_embed_dim += time_embed_dim # 768
        if self.include_spacing_input:
            self.spacing_layer = self._create_embedding_module(3, time_embed_dim)
            new_time_embed_dim += time_embed_dim # 1024

        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]
        for i in range(len(num_channels)): # [64, 128, 256, 512]
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1
            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
                num_res_blocks=num_res_blocks[i],
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
                dropout_cattn=dropout_cattn,
            )

            self.down_blocks.append(down_block)

        # mid
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=num_channels[-1],
            temb_channels=new_time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            include_fc=include_fc,
            use_combined_linear=use_combined_linear,
            use_flash_attention=use_flash_attention,
            dropout_cattn=dropout_cattn,
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(num_channels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_head_channels = list(reversed(num_head_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(num_channels) - 1)]

            is_final_block = i == len(num_channels) - 1

            up_block = get_up_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                temb_channels=new_time_embed_dim,
                num_res_blocks=reversed_num_res_blocks[i] + 1,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_upsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(reversed_attention_levels[i] and not with_conditioning),
                with_cross_attn=(reversed_attention_levels[i] and with_conditioning),
                num_head_channels=reversed_num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                include_fc=include_fc,
                use_combined_linear=use_combined_linear,
                use_flash_attention=use_flash_attention,
                dropout_cattn=dropout_cattn,
            )

            self.up_blocks.append(up_block)

        # out
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[0], eps=norm_eps, affine=True),
            nn.SiLU(),
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[0],
                    out_channels=out_channels,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            ),
        )

    def _create_embedding_module(self, input_dim, embed_dim):
        model = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        return model

    def _get_time_and_class_embedding(self, x, timesteps, class_labels):
        # num_channels[0]=64
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0]) 
        # torch.Size([1, 64])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb += class_emb
        return emb

    def _get_input_embeddings(self, emb, top_index, bottom_index, spacing):
        if self.include_top_region_index_input:
            _emb = self.top_region_index_layer(top_index)
            emb = torch.cat((emb, _emb), dim=1)
        if self.include_bottom_region_index_input:
            _emb = self.bottom_region_index_layer(bottom_index)
            emb = torch.cat((emb, _emb), dim=1)
        if self.include_spacing_input:
            _emb = self.spacing_layer(spacing)
            emb = torch.cat((emb, _emb), dim=1)
        return emb

    def _apply_down_blocks(self, h, emb, context, down_block_additional_residuals):
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            down_block_res_samples.extend(res_samples)

        # Additional residual conections for Controlnets
        if down_block_additional_residuals is not None: # 目前可忽略
            new_down_block_res_samples: list[torch.Tensor] = []
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples.append(down_block_res_sample)

            down_block_res_samples = new_down_block_res_samples
        return h, down_block_res_samples

    def _apply_up_blocks(self, h, emb, context, down_block_res_samples):
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            h = upsample_block(hidden_states=h, res_hidden_states_list=res_samples, temb=emb, context=context)

        return h

    def forward(
        self,
        x: torch.Tensor, # (N, C, SpatialDims)
        timesteps: torch.Tensor, # (N,)
        context: torch.Tensor | None = None, # (N, 1, ContextDim)
        class_labels: torch.Tensor | None = None, # (N,)
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        top_region_index_tensor: torch.Tensor | None = None, # (N, 4)
        bottom_region_index_tensor: torch.Tensor | None = None, # (N, 4)
        spacing_tensor: torch.Tensor | None = None, # (N, 3)
        layers: list=[0,0,3,6,9,11,12,13],
        encode_only: bool=False,
    ) -> torch.Tensor:
        """
        Forward pass through the UNet model.

        Args:
            x: Input tensor of shape (N, C, SpatialDims).
            timesteps: Timestep tensor of shape (N,).
            context: Context tensor of shape (N, 1, ContextDim). # 这个应该是用做text_embedding?
            class_labels: Class labels tensor of shape (N,).
            down_block_additional_residuals: Additional residual tensors for down blocks of shape (N, C, FeatureMapsDims).
            mid_block_additional_residual: Additional residual tensor for mid block of shape (N, C, FeatureMapsDims).
            top_region_index_tensor: Tensor representing top region index of shape (N, 4).
            bottom_region_index_tensor: Tensor representing bottom region index of shape (N, 4).
            spacing_tensor: Tensor representing spacing of shape (N, 3).

        Returns:
            A tensor representing the output of the UNet model.
        """
        feats=[]
        if context is not None and len(context.shape) == 2:
            context= context.unsqueeze(1)
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        # 把emb(256)和top,bottom,spacing的emb(256)拼接起来 得到new_time_embed_dim=1024
        emb = self._get_input_embeddings(emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor)
        # print('emb:',emb.shape) # torch.Size([1, 1024])
        # x: torch.Size([2, 4, 40, 40, 18])
       
        h = self.conv_in(x)
        if encode_only:
            feats.append(x)
        # h: torch.Size([2, 64, 40, 40, 18])
        h, _updated_down_block_res_samples = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        if encode_only:
            for layer_i in range(len(_updated_down_block_res_samples)):
                if layer_i in layers:
                    feats.append(_updated_down_block_res_samples[layer_i])
            feats.append(h)
        # h_down: torch.Size([2, 512, 5, 5, 18])
        '''
        down_res: 0 , torch.Size([2, 64, 40, 40, 18]) ====== h_input
        down_res: 1 , torch.Size([2, 64, 40, 40, 18]) 0 - DiffusionUNetResnetBlock
        down_res: 2 , torch.Size([2, 64, 40, 40, 18]) 0 - DiffusionUNetResnetBlock
        down_res: 3 , torch.Size([2, 64, 20, 20, 18]) 0 - DiffusionUnetDownsample
        down_res: 4 , torch.Size([2, 128, 20, 20, 18]) 1 - DiffusionUNetResnetBlock
        down_res: 5 , torch.Size([2, 128, 20, 20, 18]) 1 - DiffusionUNetResnetBlock
        down_res: 6 , torch.Size([2, 128, 10, 10, 18]) 1 - DiffusionUnetDownsample
        down_res: 7 , torch.Size([2, 256, 10, 10, 18]) 2 - resnet+ attention
        down_res: 8 , torch.Size([2, 256, 10, 10, 18]) 2 - resnet+ attention
        down_res: 9 , torch.Size([2, 256, 5, 5, 18]) 2 - DiffusionUnetDownsample
        down_res: 10 , torch.Size([2, 512, 5, 5, 18]) 3 - resnet+ attention
        down_res: 11 , torch.Size([2, 512, 5, 5, 18]) 3 - resnet+ attention
        
        '''
        h = self.middle_block(h, emb, context)
        # h_mid: torch.Size([2, 512, 5, 5, 18])
        if encode_only:
            feats.append(h)
            return feats

        # Additional residual conections for Controlnets
        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual

        h = self._apply_up_blocks(h, emb, context, _updated_down_block_res_samples)
        h = self.out(h)
        h_tensor: torch.Tensor = convert_to_tensor(h)
        return h_tensor
    
class ConvBlock_cond(nn.Module):
    def __init__(self, in_channel, out_channel, t_emb_dim, kernel_size=4, stride=1, padding=1, norm_layer=None,
                 downsample=True, use_bias=None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=use_bias)

        if norm_layer is not None:
            self.use_norm = True
            self.norm = norm_layer(out_channel)
        else:
            self.use_norm = False
        self.act = nn.LeakyReLU(0.2, True)
        # self.down = Downsample(out_channel)

        self.down = DiffusionUnetDownsample(
            spatial_dims=3,
            num_channels=out_channel,
            use_conv=True,
            out_channels=out_channel,
        )
        self.dense = nn.Linear(t_emb_dim, out_channel)

    def forward(self, input, t_emb):
        out = self.conv1(input)
        out += self.dense(t_emb)[..., None, None, None]
        if self.use_norm:
            out = self.norm(out)
        out = self.act(out)
        if self.downsample:
            out = self.down(out)

        return out

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
            # EqualLinear(hidden_dim, output_dim,bias_init = 0, activation='fused_lrelu')
        )

    def forward(self, temp):
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


class NLayerDiscriminator_ncsn(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3, norm_layer="instance", no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_ncsn, self).__init__()
        import functools
        norm_layer = functools.partial(nn.InstanceNorm3d,  eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        self.model_main = nn.ModuleList()
        kw = [4,4,1] if use144 else 4
        padw = [1,1,0] if use144 else 1
        if no_antialias:
            sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            self.model_main.append(
                ConvBlock_cond(input_nc, ndf, 4 * ndf, kernel_size=kw, stride=1, padding=padw, use_bias=use_bias))

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]
            else:
                self.model_main.append(
                    ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4 * ndf, kernel_size=kw, stride=1, padding=padw,
                                   use_bias=use_bias, norm_layer=norm_layer)

                )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4 * ndf, kernel_size=kw, stride=1, padding=padw,
                           use_bias=use_bias, norm_layer=norm_layer, downsample=False)

        )
        self.final_conv = nn.Conv3d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)
        self.t_embed = TimestepEmbedding(
            embedding_dim=4 * ndf,
            hidden_dim=4 * ndf,
            output_dim=4 * ndf,
            act=nn.LeakyReLU(0.2),
        )
        # self.embed_dim = 64
        # self.time_embed_dim = self.embed_dim * 4
        # self.t_embed = self._create_embedding_module(ndf, 4 * ndf)


    # def _create_embedding_module(self, input_dim, embed_dim):
    #     model = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
    #     return model
    
    # def _get_time_embedding(self, x, timesteps):
    #     # num_channels[0]=64
    #     t_emb = get_timestep_embedding(timesteps, self.embed_dim) 
    #     # torch.Size([1, 64])

    #     # timesteps does not contain any weights and will always return f32 tensors
    #     # but time_embedding might actually be running in fp16. so we need to cast here.
    #     # there might be better ways to encapsulate this.
    #     t_emb = t_emb.to(dtype=x.dtype)
    #     emb = self.time_embed(t_emb)

    #     return emb

    def forward(self, input, t_emb, input2=None):
        """Standard forward."""
        # [1]->[1, 256]
        t_emb = self.t_embed(t_emb)
        if input2 is not None:
            out = torch.cat([input, input2], dim=1)
        else:
            out = input
        # print('out0', out.shape)
        # out0 torch.Size([2, 16, 40, 40, 18])
        for layer in self.model_main:
            out = layer(out, t_emb)
            # print('out', out.shape)
            '''
            out torch.Size([2, 64, 20, 20, 20])
            out torch.Size([2, 128, 10, 10, 22])
            out torch.Size([2, 256, 5, 5, 24])
            out torch.Size([2, 512, 4, 4, 26])

            final torch.Size([2, 4, 3, 3, 28])

            padding改完后
            out0 torch.Size([2, 4, 40, 40, 18])
            out torch.Size([2, 64, 20, 20, 18])
            out torch.Size([2, 128, 10, 10, 18])
            out torch.Size([2, 256, 5, 5, 18])
            out torch.Size([2, 512, 4, 4, 18])
            final torch.Size([2, 4, 3, 3, 18])
            '''
        k=self.final_conv(out)
        # print('final',k.shape)
        return k

    
class Convolution_cond(nn.Sequential):
    """
    Constructs a convolution with normalization, optional dropout, and optional activation layers::

        -- (Conv|ConvTrans) -- (Norm -- Dropout -- Acti) --

    if ``conv_only`` set to ``True``::

        -- (Conv|ConvTrans) --

    For example:

    .. code-block:: python

        from monai.networks.blocks import Convolution

        conv = Convolution(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="ADN",
            act=("prelu", {"init": 0.2}),
            dropout=0.1,
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(conv)

    output::

        Convolution(
          (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
          (adn): ADN(
            (A): PReLU(num_parameters=1)
            (D): Dropout(p=0.1, inplace=False)
            (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
          )
        )

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the spatial dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zeroes out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zeroes out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no larger than the value of `spatial_dims`.
        dilation: dilation rate. Defaults to 1.
        groups: controls the connections between inputs and outputs. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        conv_only: whether to use the convolutional layer only. Defaults to False.
        is_transposed: if True uses ConvTrans instead of Conv. Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.
        output_padding: controls the additional size added to one side of the output shape.
            Defaults to None.

    See also:

        :py:class:`monai.networks.layers.Conv`
        :py:class:`monai.networks.blocks.ADN`

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.spatial_dims]

        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )

        self.add_module("conv", conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        from torch.nn import init
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, device=None, debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    
    assert(torch.cuda.is_available())
    net.to(device)
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights: # 
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

# use_mlp=True, init_type=xavier, init_gain=0.02, nc=256
class PatchSampleF_3d(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, device=None, resume=False):
        super(PatchSampleF_3d, self).__init__()
        self.l2norm = Normalize(2)  # Assuming Normalize is defined elsewhere
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.device = device
        if resume:
            self.mlp_init = True
            self.layers= [0, 0, 3, 6, 9, 11, 12, 13]
            self.input_nc=[4,64,64,128,256,512,512,512]
            for mlp_id, _ in enumerate(self.layers):
                mlp = nn.Sequential(*[nn.Linear(self.input_nc[mlp_id], self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
                setattr(self, 'mlp_%d' % mlp_id, mlp)
        

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            # if len(self.gpu_ids) > 0:
            mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.device)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        # feats: [b, c, h, w, d]
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W, D = feat.shape[0], feat.shape[2], feat.shape[3], feat.shape[4]
            feat_reshape = feat.permute(0, 2, 3, 4, 1).flatten(1, 3)  # Reshape to [B, H*W*D, C]
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # Use numpy to avoid CUDA errors with torch.randperm
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # Reshape to [B*num_patches, C]
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W, D])
            return_feats.append(x_sample)
        return return_feats, return_ids

# 未使用
class PatchDiscriminator(nn.Sequential):
    """
    Patch-GAN discriminator based on Pix2PixHD:
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs (https://arxiv.org/abs/1711.11585)


    Args:
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        channels: number of filters in the first convolutional layer (doubled for each subsequent layer)
        in_channels: number of input channels
        out_channels: number of output channels
        num_layers_d: number of Convolution layers (Conv + activation + normalisation + [dropout]) in the discriminator.
        kernel_size: kernel size of the convolution layers
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to batch norm.
        bias: whether to have a bias term in convolution blocks. Defaults to False.
        padding: padding to be applied to the convolutional layers
        dropout: proportion of dropout applied, defaults to 0.
        last_conv_kernel_size: kernel size of the last convolutional layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        channels: int,
        in_channels: int,
        out_channels: int = 1,
        num_layers_d: int = 3,
        kernel_size: int = 4,
        activation: str | tuple = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm: str | tuple = "BATCH",
        bias: bool = False,
        padding: int | Sequence[int] = 1,
        dropout: float | tuple = 0.0,
        last_conv_kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_layers_d = num_layers_d
        self.num_channels = channels
        if last_conv_kernel_size is None:
            last_conv_kernel_size = kernel_size

        self.add_module(
            "initial_conv",
            Convolution_cond( 
                # 这个是封装好的conv+act+norm+dropout, 对照着要重新写吧？
                # 如果沿用unsb的思路，就是写成conv_cond，里面会用emb，以及discrimi里面t_emb
                spatial_dims=spatial_dims,
                kernel_size=[kernel_size, kernel_size, 1] if use144 else kernel_size,
                in_channels=in_channels,
                out_channels=channels,
                act=activation,
                bias=True,
                norm=None,
                dropout=dropout,
                padding=padding,
                strides=[2,2,1] if use144 else 2,
            ),
        )

        input_channels = channels
        output_channels = channels * 2

        # Initial Layer
        for l_ in range(self.num_layers_d):
            if l_ == self.num_layers_d - 1:
                stride = 1
            else:
                stride = [2,2,1] if use144 else 2
            layer = Convolution_cond(
                spatial_dims=spatial_dims,
                kernel_size=[kernel_size, kernel_size, 1] if use144 else kernel_size,
                in_channels=input_channels,
                out_channels=output_channels,
                act=activation,
                bias=bias,
                norm=norm,
                dropout=dropout,
                padding=padding,
                strides=stride,
            )
            self.add_module("%d" % l_, layer)
            input_channels = output_channels
            output_channels = output_channels * 2

        # Final layer
        self.add_module(
            "final_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=last_conv_kernel_size,
                in_channels=input_channels,
                out_channels=out_channels,
                bias=True,
                conv_only=True,
                padding=int((last_conv_kernel_size - 1) / 2),
                dropout=0.0,
                strides=1,
            ),
        )

        self.apply(normal_init)
        self.embed_dim = 64
        self.time_embed_dim = self.embed_dim * 4
        self.time_embed = self._create_embedding_module(self.embed_dim, self.time_embed_dim)

        
        '''
Convolution_cond(
  (conv): Conv3d(1, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
  (adn): ADN(
    (D): Dropout(p=0.0, inplace=False)
    (A): LeakyReLU(negative_slope=0.2)
  )
)
Convolution_cond(
  (conv): Conv3d(32, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
  (adn): ADN(
    (N): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (D): Dropout(p=0.0, inplace=False)
    (A): LeakyReLU(negative_slope=0.2)
  )
)
Convolution_cond(
  (conv): Conv3d(64, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
  (adn): ADN(
    (N): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (D): Dropout(p=0.0, inplace=False)
    (A): LeakyReLU(negative_slope=0.2)
  )
)
Convolution_cond(
  (conv): Conv3d(128, 256, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
  (adn): ADN(
    (N): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (D): Dropout(p=0.0, inplace=False)
    (A): LeakyReLU(negative_slope=0.2)
  )
)
Convolution(
  (conv): Conv3d(256, 1, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=(1, 1, 1))
)
        
        '''
    def _create_embedding_module(self, input_dim, embed_dim):
        model = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        return model
    
    def _get_time_embedding(self, x, timesteps):
        # num_channels[0]=64
        t_emb = get_timestep_embedding(timesteps, self.embed_dim) 
        # torch.Size([1, 64])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        return emb

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, x2: torch.Tensor=None) -> list[torch.Tensor]:
        """
        Args:
            x: input tensor

        Returns:
            list of intermediate features, with the last element being the output.
        """
        
        if x2 is not None:
            x=torch.cat([x,x2],dim=1)
        out = [x]
        # 0: torch.Size([2, 4, 40, 40, 18])
        emb = self._get_time_embedding(x, timesteps)
        for submodel in self.children():
            if isinstance(submodel, Convolution_cond):
                
                dense = nn.Linear(self.time_embed_dim, submodel.conv.out_channels).to(emb.device)

                tmp = submodel.conv(out[-1])
                # tmp: torch.Size([2, 32, 20, 20, 9]) cuda:0
                tmp += dense(emb)[..., None, None, None] # 【这里不确定】[..., None, None]应该是为了格式对齐
                # a=dense(emb)[..., None, None, None]
                # a: torch.Size([1, 32, 1, 1, 1])
                # tmp +=a # ???怎么加啊
                tmp = submodel.adn(tmp)
                out.append(tmp)
                '''
                1 _cond: torch.Size([2, 64, 20, 20, 17])
                2 _cond: torch.Size([2, 128, 10, 10, 16])
                3 _cond: torch.Size([2, 256, 5, 5, 15])
                4 _cond: torch.Size([2, 512, 4, 4, 14])
                '''
            else:

                # out[-1]: torch.Size([2, 512, 4, 4, 14])
                # 下一次的时候out[-1]: torch.Size([2, 4, 3, 3, 13])然后报错
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (72x13 and 64x256)
                tmp = submodel(out[-1])
                # 5 _else: torch.Size([2, 4, 3, 3, 13])
                out.append(tmp)

        return out[1:] # 返回从第一个子模块的输出开始的所有中间输出
    

if __name__ == "__main__":
    model=PatchDiscriminator(
        spatial_dims=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        num_layers_d=3,
        norm="INSTANCE",
    )
    input=torch.randn(1,1,64,64,64)
    timesteps=torch.randn(1)
    output=model(input,timesteps)

    # netG=Generator(
    #     spatial_dims=3,
    #     in_channels=4,
    #     out_channels=4,
    # )
    # input=torch.randn(1,4,64,64,64)
    # timesteps=torch.randn(1)
    # output=netG(input,timesteps)


