""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""

# Copyright 2026 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on pytorch-image-models (timm); see NOTICE.
#
# Modifications:
# - Added nested progressive subnetworks, progressive resolution,
#   representation reuse, and progress-conditioned soft gating.

import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from torch import Tensor, Size
from torch.nn.parameter import Parameter
from torch.nn import init

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, Format, nchw_to, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.layers.trace_utils import _assert
from ._builder import build_model_with_cfg
from ._manipulate import named_apply, adapt_input_conv
from ._registry import generate_default_cfgs, register_model as _register_model

__all__ = ['ProgResViT', 'VisionTransformer']
_shape_t = Union[int, List[int], Size]

_logger = logging.getLogger(__name__)


def register_model(fn):
    """Register only the public ProgResViT entry point from this vendored module."""
    return _register_model(fn) if fn.__name__ == 'progresvit' else fn


def register_model_deprecations(*args, **kwargs):
    """Do not expose upstream ViT aliases that are unsupported by this release."""
    return None


def _resolve_active_dim(active_dim: Optional[int], full_dim: int) -> int:
    return int(full_dim if active_dim is None else active_dim)


def _scale_active_dim(active_dim: int, source_dim: int, target_dim: int) -> int:
    if source_dim == target_dim:
        return int(active_dim)
    return int(active_dim) * int(target_dim) // int(source_dim)


class Identity(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: Tensor, active_dim: Optional[int] = None) -> Tensor:
        return input

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x, active_dim: Optional[int] = None):
        active_dim_in = _resolve_active_dim(active_dim, self.in_features)
        active_dim_hidden = _scale_active_dim(active_dim_in, self.in_features, self.hidden_features)
        active_dim_out = _scale_active_dim(active_dim_in, self.in_features, self.out_features)

        x = self.fc1(x, active_dim_in, active_dim_hidden)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, active_dim_hidden, active_dim_out)
        x = self.drop(x)
        return x


class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, p=float) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(
            self,
            input: Tensor,
            active_dim_in: Optional[int] = None,
            active_dim_out: Optional[int] = None,
    ) -> Tensor:
        active_dim_in = _resolve_active_dim(active_dim_in, self.in_features)
        active_dim_out = active_dim_in if active_dim_out is None else int(active_dim_out)

        if active_dim_in >= self.in_features and active_dim_out >= self.out_features:
            return F.linear(input, self.weight, self.bias)

        bias = None if self.bias is None else self.bias[:active_dim_out]
        return F.linear(
            input[..., :active_dim_in],
            self.weight[:active_dim_out, :active_dim_in],
            bias,
        )

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class QKVLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, p=float) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, active_dim: Optional[int] = None) -> Tensor:
        active_dim = _resolve_active_dim(active_dim, self.in_features)
        if active_dim >= self.in_features:
            return F.linear(input, self.weight, self.bias)

        qkv_dim = self.out_features // 3
        weight = self.weight.view(3, qkv_dim, self.in_features)
        weight = weight[:, :active_dim, :active_dim].reshape(3 * active_dim, active_dim)
        bias = None
        if self.bias is not None:
            bias = self.bias.view(3, qkv_dim)[:, :active_dim].reshape(3 * active_dim)

        return F.linear(input[..., :active_dim], weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class LayerNorm(nn.Module):

    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor, active_dim: Optional[int] = None) -> Tensor:
        active_dim = _resolve_active_dim(active_dim, self.normalized_shape[0])
        if active_dim >= self.normalized_shape[0]:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

        return F.layer_norm(
            input[..., :active_dim],
            (active_dim,),
            self.weight[:active_dim],
            self.bias[:active_dim],
            self.eps,
        )

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)



class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = QKVLinear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dim = dim
    def forward(
            self,
            x: torch.Tensor,
            active_dim: Optional[int] = None,
            active_heads: Optional[int] = None,
        ) -> torch.Tensor:
        B, N, C = x.shape
        active_dim = _resolve_active_dim(active_dim, self.dim)
        active_heads = active_dim // self.head_dim if active_heads is None else int(active_heads)

        qkv = self.qkv(x, active_dim).reshape(
            B, N, 3, active_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, active_dim)
        x = self.proj(x, active_dim)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x, active_dim: Optional[int] = None):
        active_dim = _resolve_active_dim(active_dim, self.gamma.shape[0])
        gamma = self.gamma[:active_dim]
        return x.mul_(gamma) if self.inplace else x * gamma


class CondMLP(nn.Module):
    def __init__(self, in_features: int = 5, hidden_features: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CondGate(nn.Module):
    """Conditioned multiplier initialized as an identity gate.

    The small MLP predicts a delta whose final layer starts at zero. Returning
    ``1 + delta`` here keeps the residual/fusion multiplier convention in one
    place and preserves normal ViT behavior at initialization.
    """

    def __init__(self, cond_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_features, cond_features),
            nn.SiLU(),
            nn.Linear(cond_features, out_features),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond_emb: torch.Tensor, active_dim: Optional[int] = None) -> torch.Tensor:
        delta = self.net(cond_emb)
        if active_dim is not None:
            delta = delta[:, :active_dim]
        return (1. + delta).unsqueeze(1)



class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=LayerNorm,
            mlp_layer=Mlp,
            loop_cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.dim = dim
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gate_attn = CondGate(loop_cond_dim, dim) if loop_cond_dim is not None else None
        self.gate_mlp = CondGate(loop_cond_dim, dim) if loop_cond_dim is not None else None
        self.gate_out = CondGate(loop_cond_dim, dim) if loop_cond_dim is not None else None
        self.prog_ls1 = LayerScale(dim, init_values=1.0) if loop_cond_dim is not None else Identity()
        self.prog_ls2 = LayerScale(dim, init_values=1.0) if loop_cond_dim is not None else Identity()

    def forward(
            self,
            x: torch.Tensor,
            active_dim: Optional[int] = None,
            active_heads: Optional[int] = None,
            cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        active_dim = _resolve_active_dim(active_dim, self.dim)
        attn_out = self.attn(self.norm1(x, active_dim), active_dim, active_heads)
        attn_out = self.ls1(attn_out, active_dim)
        attn_out = self.prog_ls1(attn_out, active_dim)
        if cond_emb is not None and self.gate_attn is not None:
            attn_out = self.gate_attn(cond_emb, active_dim) * attn_out
        x = x[:, :, :active_dim] + self.drop_path1(attn_out)

        mlp_out = self.mlp(self.norm2(x, active_dim), active_dim)
        mlp_out = self.ls2(mlp_out, active_dim)
        mlp_out = self.prog_ls2(mlp_out, active_dim)
        if cond_emb is not None and self.gate_mlp is not None:
            mlp_out = self.gate_mlp(cond_emb, active_dim) * mlp_out
        x = x[:, :, :active_dim] + self.drop_path2(mlp_out)
        if cond_emb is not None and self.gate_out is not None:
            x = self.gate_out(cond_emb, active_dim) * x
        return x


def _to_2tuple_int(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 2:
            raise ValueError(f'Patch size must be an int or a 2-tuple, got {value}.')
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _default_progress_img_sizes(
        num_stages: int,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
) -> Tuple[int, ...]:
    full_size = int(img_size[0])
    if img_size[0] != img_size[1]:
        raise ValueError('Default progress_img_sizes only supports square img_size.')
    if patch_size[0] != patch_size[1]:
        raise ValueError('Default progress_img_sizes only supports square patch_size.')
    full_grid = full_size // int(patch_size[0])

    def grid_to_img(grid: int) -> int:
        return int(grid) * int(patch_size[0])

    if num_stages == 2:
        return grid_to_img(max(1, full_grid * 4 // 5)), full_size
    if num_stages == 3:
        return grid_to_img(max(1, full_grid * 2 // 3)), grid_to_img(max(1, full_grid * 4 // 5)), full_size
    raise ValueError(f'progress_stages must contain 2 or 3 stages, got {num_stages}')


class TokenProjector(nn.Module):
    """Project previous-stage ViT tokens onto the current-stage token grid."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            source_hw: Tuple[int, int],
            target_hw: Tuple[int, int],
            num_prefix_tokens: int = 1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.source_hw = tuple(int(v) for v in source_hw)
        self.target_hw = tuple(int(v) for v in target_hw)
        self.num_prefix_tokens = int(num_prefix_tokens)
        if self.in_channels < 1 or self.out_channels < 1:
            raise ValueError('Projector channel counts must be positive.')

        self.depthwise_proj = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=3, padding=1,
            groups=self.in_channels, bias=False)
        self._init_depthwise_identity()
        self.spatial_proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
        self.prefix_proj = nn.Linear(self.in_channels, self.out_channels) if self.num_prefix_tokens else None
        self._init_channel_projection_identity()

    def _init_depthwise_identity(self) -> None:
        with torch.no_grad():
            self.depthwise_proj.weight.zero_()
            self.depthwise_proj.weight[:, 0, 1, 1].fill_(1.)

    def _init_channel_projection_identity(self) -> None:
        """Copy shared channels and zero-fill new channels at initialization."""
        shared_channels = min(self.in_channels, self.out_channels)
        with torch.no_grad():
            self.spatial_proj.weight.zero_()
            self.spatial_proj.bias.zero_()
            for idx in range(shared_channels):
                self.spatial_proj.weight[idx, idx, 0, 0] = 1.

            if self.prefix_proj is not None:
                self.prefix_proj.weight.zero_()
                self.prefix_proj.bias.zero_()
                for idx in range(shared_channels):
                    self.prefix_proj.weight[idx, idx] = 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.num_prefix_tokens + self.source_hw[0] * self.source_hw[1]
        if x.shape[1] != expected_tokens:
            raise ValueError(
                f'Projector expected {expected_tokens} tokens for grid {self.source_hw}, got {x.shape[1]}.'
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f'Projector was built for {self.in_channels} input channels, got {x.shape[-1]}.'
            )

        prefix = x[:, :self.num_prefix_tokens] if self.num_prefix_tokens else None
        spatial = x[:, self.num_prefix_tokens:]
        spatial = spatial.reshape(x.shape[0], self.source_hw[0], self.source_hw[1], self.in_channels)
        spatial = spatial.permute(0, 3, 1, 2)
        if self.source_hw != self.target_hw:
            spatial = F.interpolate(spatial, size=self.target_hw, mode='bilinear', align_corners=False)
        spatial = self.depthwise_proj(spatial)
        spatial = self.spatial_proj(spatial)
        spatial = spatial.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.out_channels)

        if prefix is None:
            return spatial
        prefix = self.prefix_proj(prefix)
        return torch.cat([prefix, spatial], dim=1)


class ResPostBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelScalingBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class ParallelThingsBlock(nn.Module):
    """ Parallel ViT block (N parallel attention followed by N parallel MLP)
    Based on:
      `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_parallel: int = 2,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                )),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', mlp_layer(
                    dim,
                    hidden_features=int(dim * mlp_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


class ProgResViT(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            progress_stages: Optional[Sequence[int]] = None,
            progress_img_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU
        assert embed_dim % num_heads == 0, 'embed_dim should be divisible by num_heads'

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.img_size = _to_2tuple_int(img_size) if img_size is not None else None

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self._pos_embed_eval_cache = {}
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        loop_cond_dim = 128 if progress_stages is not None else None
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_kwargs = dict(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            if block_fn is Block:
                block_kwargs['loop_cond_dim'] = loop_cond_dim
            self.blocks.append(block_fn(**block_kwargs))
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

        self.progress_stages = tuple(int(s) for s in progress_stages) if progress_stages is not None else ()
        base_patch_hw = _to_2tuple_int(self.patch_embed.patch_size)
        self.progress_patch_size = base_patch_hw[0] if base_patch_hw[0] == base_patch_hw[1] else base_patch_hw
        if progress_img_sizes is None:
            progress_img_sizes = (
                _default_progress_img_sizes(len(self.progress_stages), self.img_size, base_patch_hw)
                if self.progress_stages else ()
            )
        self.progress_img_sizes = tuple(int(s) for s in progress_img_sizes)
        if self.progress_stages and len(self.progress_img_sizes) != len(self.progress_stages):
            raise ValueError(
                f'progress_img_sizes must match progress_stages length: '
                f'{self.progress_img_sizes} vs {self.progress_stages}'
            )
        for img_size in self.progress_img_sizes:
            if self.img_size is None:
                raise ValueError('progress_img_sizes requires a fixed img_size.')
            img_hw = _to_2tuple_int(img_size)
            if img_hw[0] > self.img_size[0] or img_hw[1] > self.img_size[1]:
                raise ValueError(f'progress_img_size {img_hw} cannot exceed model img_size {self.img_size}.')
            if img_hw[0] % base_patch_hw[0] != 0 or img_hw[1] % base_patch_hw[1] != 0:
                raise ValueError(f'progress_img_size {img_hw} must be divisible by patch size {base_patch_hw}.')
        self.loop_cond_dim = loop_cond_dim
        self.cond_mlp = CondMLP(5, loop_cond_dim) if loop_cond_dim is not None else None
        self.token_projectors = nn.ModuleList()
        self.fusion_input_delta = nn.ModuleList()
        self.fusion_prev_delta = nn.ModuleList()
        for src_idx in range(max(0, len(self.progress_stages) - 1)):
            out_channels = self._stage_dim(self.progress_stages[src_idx + 1])
            self.token_projectors.append(TokenProjector(
                in_channels=self._stage_dim(self.progress_stages[src_idx]),
                out_channels=out_channels,
                source_hw=self._grid_size_for_img_size(
                    self.progress_img_sizes[src_idx], base_patch_hw),
                target_hw=self._grid_size_for_img_size(
                    self.progress_img_sizes[src_idx + 1], base_patch_hw),
                num_prefix_tokens=self.num_prefix_tokens,
            ))
            self.fusion_input_delta.append(CondGate(loop_cond_dim, out_channels))
            self.fusion_prev_delta.append(CondGate(loop_cond_dim, out_channels))
        self._zero_init_condition_gate_heads()

    def _resolve_progress_stages(self, progress_stages=None) -> Tuple[int, ...]:
        stages = progress_stages if progress_stages is not None else self.progress_stages
        stages = tuple(int(s) for s in stages)
        if len(stages) not in (2, 3):
            raise ValueError(f'progress_stages must contain 2 or 3 stages, got {stages}')
        if max(stages) > self.num_heads:
            raise ValueError(
                f'progress_stages max ({max(stages)}) cannot exceed model num_heads ({self.num_heads})'
            )
        return stages

    def _resolve_progress_img_sizes(self, progress_stages=None, progress_img_sizes=None) -> Tuple[int, ...]:
        stages = self._resolve_progress_stages(progress_stages)
        img_sizes = progress_img_sizes if progress_img_sizes is not None else self.progress_img_sizes
        img_sizes = tuple(int(s) for s in img_sizes)
        base_patch_hw = _to_2tuple_int(self.patch_embed.patch_size)
        if not img_sizes:
            img_sizes = _default_progress_img_sizes(len(stages), self.img_size, base_patch_hw)
        if len(img_sizes) != len(stages):
            raise ValueError(f'progress_img_sizes must match progress_stages: {img_sizes} vs {stages}')
        for img_size in img_sizes:
            if self.img_size is None:
                raise ValueError('progress_img_sizes requires a fixed img_size.')
            img_hw = _to_2tuple_int(img_size)
            if img_hw[0] > self.img_size[0] or img_hw[1] > self.img_size[1]:
                raise ValueError(f'progress_img_size {img_hw} cannot exceed model img_size {self.img_size}.')
            if img_hw[0] % base_patch_hw[0] != 0 or img_hw[1] % base_patch_hw[1] != 0:
                raise ValueError(f'progress_img_size {img_hw} must be divisible by patch size {base_patch_hw}.')
        return img_sizes

    def _stage_dim(self, stage_heads: int) -> int:
        return int(stage_heads) * self.head_dim

    def _grid_size_for_img_size(
            self,
            img_size: Union[int, Sequence[int]],
            patch_size: Union[int, Sequence[int]],
    ) -> Tuple[int, int]:
        img_hw = _to_2tuple_int(img_size)
        patch_hw = _to_2tuple_int(patch_size)
        return img_hw[0] // patch_hw[0], img_hw[1] // patch_hw[1]

    def _zero_init_condition_gate_heads(self) -> None:
        for module in self.modules():
            if isinstance(module, CondGate):
                nn.init.zeros_(module.net[-1].weight)
                nn.init.zeros_(module.net[-1].bias)

    def _condition_embedding(
            self,
            stage_idx: int,
            block_idx: int,
            progress_img_sizes: Sequence[int],
            ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.cond_mlp is None:
            return None
        depth = len(self.blocks)
        total_calls = max(1, len(progress_img_sizes) * depth - 1)
        global_progress = (stage_idx * depth + block_idx) / total_calls
        cur_res = float(_to_2tuple_int(progress_img_sizes[stage_idx])[0])
        prev_res = cur_res if stage_idx == 0 else float(_to_2tuple_int(progress_img_sizes[stage_idx - 1])[0])
        cond = ref.new_tensor([[
            float(stage_idx),
            float(global_progress),
            math.log2(cur_res / 224.),
            math.log2(prev_res / 224.),
            math.log2(cur_res / prev_res),
        ]])
        return self.cond_mlp(cond)

    def _condition_embeddings(
            self,
            stage_idx: int,
            progress_img_sizes: Sequence[int],
            ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.cond_mlp is None:
            return None

        depth = len(self.blocks)
        total_calls = max(1, len(progress_img_sizes) * depth - 1)
        cur_res = float(_to_2tuple_int(progress_img_sizes[stage_idx])[0])
        prev_res = cur_res if stage_idx == 0 else float(_to_2tuple_int(progress_img_sizes[stage_idx - 1])[0])
        rows = [
            [
                float(stage_idx),
                float((stage_idx * depth + block_idx) / total_calls),
                math.log2(cur_res / 224.),
                math.log2(prev_res / 224.),
                math.log2(cur_res / prev_res),
            ]
            for block_idx in range(depth)
        ]
        return self.cond_mlp(ref.new_tensor(rows))

    def _pos_embed_for_grid(self, grid_size: Tuple[int, int]) -> torch.Tensor:
        num_prefix_tokens = 0 if self.no_embed_class else self.num_prefix_tokens
        if self.training or torch.is_grad_enabled():
            return resample_abs_pos_embed(
                self.pos_embed,
                grid_size,
                old_size=self.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
            )

        cache_key = (
            tuple(int(v) for v in grid_size),
            str(self.pos_embed.device),
            str(self.pos_embed.dtype),
            self.pos_embed._version,
        )
        pos_embed = self._pos_embed_eval_cache.get(cache_key)
        if pos_embed is None:
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                grid_size,
                old_size=self.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
            )
            self._pos_embed_eval_cache[cache_key] = pos_embed
            if len(self._pos_embed_eval_cache) > 8:
                self._pos_embed_eval_cache.pop(next(iter(self._pos_embed_eval_cache)))
        return pos_embed

    def _patch_embed_with_img_size(
            self,
            x: torch.Tensor,
            img_size: Union[int, Sequence[int]],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        img_hw = _to_2tuple_int(img_size)
        patch_hw = _to_2tuple_int(self.patch_embed.patch_size)
        _, _, H, W = x.shape
        if (H, W) != img_hw:
            x = F.interpolate(x, size=img_hw, mode='bicubic', align_corners=False, antialias=True)
            H, W = img_hw
        if self.patch_embed.dynamic_img_pad:
            pad_h = (patch_hw[0] - H % patch_hw[0]) % patch_hw[0]
            pad_w = (patch_hw[1] - W % patch_hw[1]) % patch_hw[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            _assert(H % patch_hw[0] == 0, f"Input height ({H}) should be divisible by patch size ({patch_hw[0]}).")
            _assert(W % patch_hw[1] == 0, f"Input width ({W}) should be divisible by patch size ({patch_hw[1]}).")

        x = F.conv2d(x, self.patch_embed.proj.weight, self.patch_embed.proj.bias, stride=patch_hw)
        grid_size = (x.shape[-2], x.shape[-1])
        if self.patch_embed.flatten:
            x = x.flatten(2).transpose(1, 2)
        elif self.patch_embed.output_fmt != Format.NCHW:
            x = nchw_to(x, self.patch_embed.output_fmt)
        x = self.patch_embed.norm(x)
        return x, grid_size

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def _pos_embed(self, x: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if grid_size is not None:
            pos_embed = self._pos_embed_for_grid(grid_size)
        elif self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(
            self,
            x: torch.Tensor,
            active_dim: int,
            active_heads: int,
            x_out_prev=None,
            stage_idx: int = 0,
            progress_img_sizes: Sequence[int] = (),
    ) -> torch.Tensor:
        x, grid_size = self._patch_embed_with_img_size(x, progress_img_sizes[stage_idx])
        x = self._pos_embed(x, grid_size=grid_size)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = x[:, :, :active_dim]
        cond_embs = self._condition_embeddings(stage_idx, progress_img_sizes, x)

        if x_out_prev is not None:
            projector_idx = stage_idx - 1
            if projector_idx < 0 or projector_idx >= len(self.token_projectors):
                raise ValueError(f'No token projector available for progress stage {stage_idx}.')
            projected_prev = self.token_projectors[projector_idx](x_out_prev)
            cond_emb = None if cond_embs is None else cond_embs[0:1]
            input_gate = self.fusion_input_delta[projector_idx](cond_emb, active_dim)
            prev_gate = self.fusion_prev_delta[projector_idx](cond_emb, active_dim)
            x = input_gate * x + prev_gate * projected_prev

        for i, blk in enumerate(self.blocks):
            cond_emb = None if cond_embs is None else cond_embs[i:i + 1]
            x = blk(x, active_dim, active_heads, cond_emb)
        x = self.norm(x, active_dim)
        return x

    def entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        topk_count = min(10, probs.shape[1])
        topk = torch.topk(probs, topk_count, dim=1).values
        topk = topk / topk.sum(dim=1, keepdim=True).clamp_min(torch.finfo(topk.dtype).tiny)
        return -(topk * topk.clamp_min(torch.finfo(topk.dtype).tiny).log()).sum(dim=1, keepdim=True)

    def _forward_stage(
            self,
            x: torch.Tensor,
            stage_idx: int,
            x_out_prev: Optional[torch.Tensor],
            progress_stages: Tuple[int, ...],
            progress_img_sizes: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        active_heads = int(progress_stages[stage_idx])
        active_dim = self._stage_dim(active_heads)
        tokens = self.forward_features(
            x,
            active_dim,
            active_heads,
            x_out_prev=x_out_prev,
            stage_idx=stage_idx,
            progress_img_sizes=progress_img_sizes,
        )
        return tokens, self.forward_head(tokens, active_dim=active_dim)

    def _forward_stage_sequence(
            self,
            x: torch.Tensor,
            progress_stages: Tuple[int, ...],
            progress_img_sizes: Tuple[int, ...],
            stop_stage_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        logits = []
        x_out_prev = None
        last_stage_idx = len(progress_stages) - 1 if stop_stage_idx is None else int(stop_stage_idx)
        for stage_idx in range(last_stage_idx + 1):
            x_out_prev, stage_logits = self._forward_stage(
                x, stage_idx, x_out_prev, progress_stages, progress_img_sizes)
            logits.append(stage_logits)
        return tuple(logits)

    def _forward_threshold(
            self,
            x: torch.Tensor,
            threshold,
            progress_stages: Tuple[int, ...],
            progress_img_sizes: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if threshold is None:
            logits = self._forward_stage_sequence(x, progress_stages, progress_img_sizes)
            mask = torch.full(
                (x.shape[0], 1),
                len(progress_stages) - 1,
                dtype=torch.long,
                device=x.device,
            )
            return logits[-1], mask

        active_x = x
        active_indices = torch.arange(x.shape[0], device=x.device)
        x_out_prev = None
        output_logits = None
        mask = torch.full(
            (x.shape[0], 1),
            len(progress_stages) - 1,
            dtype=torch.long,
            device=x.device,
        )

        for stage_idx in range(len(progress_stages)):
            x_out, logits = self._forward_stage(
                active_x, stage_idx, x_out_prev, progress_stages, progress_img_sizes)
            if output_logits is None:
                output_logits = logits.new_empty(x.shape[0], logits.shape[-1])

            is_last_stage = stage_idx == len(progress_stages) - 1
            if is_last_stage:
                output_logits[active_indices] = logits
                mask[active_indices, 0] = stage_idx
                break

            exit_now = (self.entropy(logits).squeeze(1) < threshold)
            if exit_now.any():
                output_logits[active_indices[exit_now]] = logits[exit_now]
                mask[active_indices[exit_now], 0] = stage_idx

            keep_going = ~exit_now
            if not keep_going.any():
                break

            # Only uncertain samples continue. Their previous-stage tokens are
            # kept aligned with the sliced input batch for token recycling.
            active_x = active_x[keep_going]
            active_indices = active_indices[keep_going]
            x_out_prev = x_out[keep_going]

        return output_logits, mask


    def forward(
            self,
            x: torch.Tensor,
            threshold=None,
            progress_stages=None,
            progress_img_sizes=None,
            eval_stage_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.progress_stages:
            if progress_stages is not None or progress_img_sizes is not None:
                raise ValueError('Progressive arguments require a model initialized with progress_stages.')
            if threshold is not None or eval_stage_idx is not None:
                raise ValueError('Routing arguments require a model initialized with progress_stages.')
            input_size = (int(x.shape[-2]), int(x.shape[-1]))
            tokens = self.forward_features(
                x,
                self.embed_dim,
                self.num_heads,
                stage_idx=0,
                progress_img_sizes=(input_size,),
            )
            return self.forward_head(tokens)

        progress_stages = self._resolve_progress_stages(progress_stages)
        progress_img_sizes = self._resolve_progress_img_sizes(progress_stages, progress_img_sizes)
        if self.training:
            return self._forward_stage_sequence(x, progress_stages, progress_img_sizes)

        if eval_stage_idx is not None:
            stage_idx = int(eval_stage_idx)
            if stage_idx < 0 or stage_idx >= len(progress_stages):
                raise ValueError(f'eval_stage_idx {stage_idx} is invalid for {len(progress_stages)} stages.')
            return self._forward_stage_sequence(
                x, progress_stages, progress_img_sizes, stop_stage_idx=stage_idx)[-1]

        return self._forward_threshold(x, threshold, progress_stages, progress_img_sizes)

    def forward_head(
            self,
            x: torch.Tensor,
            active_dim: Optional[int] = None,
            pre_logits: bool = False,
    ) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)

        active_dim = self.embed_dim if active_dim is None else int(active_dim)
        if pre_logits or isinstance(self.head, nn.Identity):
            return x
        return self.head(x, active_dim, self.num_classes)


# Backwards compatibility for existing timm/DeiT imports and old user scripts.
VisionTransformer = ProgResViT

def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> Callable:
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def _load_weights(model: ProgResViT, checkpoint_path: str, prefix: str = '') -> None:
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))


def _convert_openai_clip(
        state_dict: Dict[str, torch.Tensor],
        model: ProgResViT,
        prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                    model.patch_embed.grid_size
                )
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
        state_dict: Dict[str, torch.Tensor],
        model: ProgResViT,
) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: ProgResViT,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model, prefix='module.visual.')

    if "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)

    if "encoder" in state_dict:
        state_dict = state_dict['encoder']
        prefix = 'module.'

    if 'visual.trunk.pos_embed' in state_dict:
        # convert an OpenCLIP model with timm vision encoder
        # FIXME remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
        prefix = 'visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }

default_cfgs = {
    'progresvit': _cfg(input_size=(3, 240, 240), mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),

    # re-finetuned augreg 21k FT on in1k weights
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/'),
    'vit_base_patch16_384.augreg2_in21k_ft_in1k': _cfg(),
    'vit_base_patch8_224.augreg2_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/'),

    # How to train your ViT (augreg) weights, pretrained on 21k FT on in1k
    'vit_tiny_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_tiny_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_large_patch16_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_large_patch16_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),

    # patch models (weights from official Google JAX impl) pretrained on in21k FT on in1k
    'vit_base_patch16_224.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        hf_hub_id='timm/'),
    'vit_base_patch16_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),

    # How to train your ViT (augreg) weights trained on in1k only
    'vit_small_patch16_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_small_patch16_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch32_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True),
    'vit_base_patch16_384.augreg_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        custom_load=True, input_size=(3, 384, 384), crop_pct=1.0),

    'vit_large_patch14_224.untrained': _cfg(url=''),
    'vit_huge_patch14_224.untrained': _cfg(url=''),
    'vit_giant_patch14_224.untrained': _cfg(url=''),
    'vit_gigantic_patch14_224.untrained': _cfg(url=''),

    # patch models, imagenet21k (weights from official Google JAX impl), classifier not valid
    'vit_base_patch32_224.orig_in21k': _cfg(
        #url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        hf_hub_id='timm/',
        num_classes=0),
    'vit_base_patch16_224.orig_in21k': _cfg(
        #url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        hf_hub_id='timm/',
        num_classes=0),
    'vit_large_patch32_224.orig_in21k': _cfg(
        #url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        hf_hub_id='timm/',
        num_classes=0),
    'vit_large_patch16_224.orig_in21k': _cfg(
        #url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        hf_hub_id='timm/',
        num_classes=0),
    'vit_huge_patch14_224.orig_in21k': _cfg(
        hf_hub_id='timm/',
        num_classes=0),

    # How to train your ViT (augreg) weights, pretrained on in21k
    'vit_tiny_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_small_patch32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_small_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_base_patch8_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),
    'vit_large_patch16_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        hf_hub_id='timm/',
        custom_load=True, num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_224.sam_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz', custom_load=True,
        hf_hub_id='timm/'),
    'vit_base_patch16_224.sam_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz', custom_load=True,
        hf_hub_id='timm/'),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'vit_small_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch8_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch16_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch8_224.dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        hf_hub_id='timm/',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    # DINOv2 pretrained - https://arxiv.org/abs/2304.07193 (no classifier head, for fine-tune/features only)
    'vit_small_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_base_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_large_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_giant_patch14_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # DINOv2 pretrained w/ registers - https://arxiv.org/abs/2309.16588 (no classifier head, for fine-tune/features only)
    'vit_small_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_base_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_large_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),
    'vit_giant_patch14_reg4_dinov2.lvd142m': _cfg(
        url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth',
        hf_hub_id='timm/',
        license='apache-2.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0,
        input_size=(3, 518, 518), crop_pct=1.0),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil.in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_in21k_miil-887286df.pth',
        hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear', num_classes=11221),
    'vit_base_patch16_224_miil.in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/vit_base_patch16_224_1k_miil_84_4-2deb18e3.pth',
        hf_hub_id='timm/',
        mean=(0., 0., 0.), std=(1., 1., 1.), crop_pct=0.875, interpolation='bilinear'),

    # Custom timm variants
    'vit_base_patch16_rpn_224.sw_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/vit_base_patch16_rpn_224-sw-3b07e89d.pth',
        hf_hub_id='timm/'),
    'vit_medium_patch16_gap_240.sw_in12k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=11821),
    'vit_medium_patch16_gap_256.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), crop_pct=0.95),
    'vit_medium_patch16_gap_384.sw_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=0.95, crop_mode='squash'),
    'vit_base_patch16_gap_224': _cfg(),

    # CLIP pretrained image tower and related fine-tuned weights
    'vit_base_patch32_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch32_clip_384.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, input_size=(3, 384, 384)),
    'vit_base_patch32_clip_448.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, input_size=(3, 448, 448)),
    'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=0.95),
    'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),
    'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.openai_ft_in12k_in1k': _cfg(
        # hf_hub_id='timm/vit_base_patch32_clip_224.openai_ft_in12k_in1k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch32_clip_384.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=0.95, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_base_patch16_clip_224.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=0.95),
    'vit_base_patch16_clip_384.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=0.95, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.openai_ft_in12k_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_base_patch16_clip_384.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0),
    'vit_large_patch14_clip_336.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),
    'vit_huge_patch14_clip_224.laion2b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),
    'vit_huge_patch14_clip_336.laion2b_ft_in1k': _cfg(
        hf_hub_id='',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), crop_mode='squash'),

    'vit_base_patch32_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    'vit_base_patch16_clip_384.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 384, 384), crop_mode='squash'),
    'vit_large_patch14_clip_224.openai_ft_in1k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0),

    'vit_base_patch32_clip_224.laion2b_ft_in12k': _cfg(
        #hf_hub_id='timm/vit_base_patch32_clip_224.laion2b_ft_in12k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_base_patch16_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_large_patch14_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0, num_classes=11821),
    'vit_huge_patch14_clip_224.laion2b_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=11821),

    'vit_base_patch32_clip_224.openai_ft_in12k': _cfg(
        # hf_hub_id='timm/vit_base_patch32_clip_224.openai_ft_in12k',  # FIXME weight exists, need to push
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_base_patch16_clip_224.openai_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=11821),
    'vit_large_patch14_clip_224.openai_ft_in12k': _cfg(
        hf_hub_id='timm/',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=11821),

    'vit_base_patch32_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_base_patch16_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-16-laion2B-s34B-b88K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, crop_pct=1.0, num_classes=768),
    'vit_huge_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_giant_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-g-14-laion2B-s12B-b42K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_gigantic_patch14_clip_224.laion2b': _cfg(
        hf_hub_id='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1280),

    'vit_base_patch32_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_base_patch32_clip_256.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 256, 256), num_classes=512),
    'vit_base_patch16_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.datacompxl': _cfg(
        hf_hub_id='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),

    'vit_base_patch16_clip_224.dfn2b': _cfg(
        hf_hub_id='apple/DFN2B-CLIP-ViT-B-16',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.dfn2b': _cfg(
        hf_hub_id='apple/DFN2B-CLIP-ViT-L-14',
        hf_hub_filename='open_clip_pytorch_model.bin',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_huge_patch14_clip_224.dfn5b': _cfg(
        hf_hub_id='apple/DFN5B-CLIP-ViT-H-14',
        hf_hub_filename='open_clip_pytorch_model.bin',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),
    'vit_huge_patch14_clip_378.dfn5b': _cfg(
        hf_hub_id='apple/DFN5B-CLIP-ViT-H-14-378',
        hf_hub_filename='open_clip_pytorch_model.bin',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        crop_pct=1.0, input_size=(3, 378, 378), num_classes=1024),

    'vit_base_patch32_clip_224.metaclip_2pt5b': _cfg(
        hf_hub_id='facebook/metaclip-b32-fullcc2.5b',
        hf_hub_filename='metaclip_b32_fullcc2.5b.bin',
        license='cc-by-nc-4.0',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_base_patch16_clip_224.metaclip_2pt5b': _cfg(
        hf_hub_id='facebook/metaclip-b16-fullcc2.5b',
        hf_hub_filename='metaclip_b16_fullcc2.5b.bin',
        license='cc-by-nc-4.0',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=512),
    'vit_large_patch14_clip_224.metaclip_2pt5b': _cfg(
        hf_hub_id='facebook/metaclip-l14-fullcc2.5b',
        hf_hub_filename='metaclip_l14_fullcc2.5b.bin',
        license='cc-by-nc-4.0',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_huge_patch14_clip_224.metaclip_2pt5b': _cfg(
        hf_hub_id='facebook/metaclip-h14-fullcc2.5b',
        hf_hub_filename='metaclip_h14_fullcc2.5b.bin',
        license='cc-by-nc-4.0',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=1024),

    'vit_base_patch32_clip_224.openai': _cfg(
        hf_hub_id='timm/vit_base_patch32_clip_224.openai',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_base_patch16_clip_224.openai': _cfg(
        hf_hub_id='timm/vit_base_patch16_clip_224.openai',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, num_classes=512),
    'vit_large_patch14_clip_224.openai': _cfg(
        hf_hub_id='timm/vit_large_patch14_clip_224.openai',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, crop_pct=1.0, num_classes=768),
    'vit_large_patch14_clip_336.openai': _cfg(
        hf_hub_id='timm/vit_large_patch14_clip_336.openai', hf_hub_filename='open_clip_pytorch_model.bin',
        notes=('natively QuickGELU, use quickgelu model variant for original results',),
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        crop_pct=1.0, input_size=(3, 336, 336), num_classes=768),

    # experimental (may be removed)
    'vit_base_patch32_plus_256.untrained': _cfg(url='', input_size=(3, 256, 256), crop_pct=0.95),
    'vit_base_patch16_plus_240.untrained': _cfg(url='', input_size=(3, 240, 240), crop_pct=0.95),
    'vit_small_patch16_36x1_224.untrained': _cfg(url=''),
    'vit_small_patch16_18x2_224.untrained': _cfg(url=''),
    'vit_base_patch16_18x2_224.untrained': _cfg(url=''),

    # EVA fine-tuned weights from MAE style MIM - EVA-CLIP target pretrain
    # https://github.com/baaivision/EVA/blob/7ecf2c0a370d97967e86d047d7af9188f78d2df3/eva/README.md#eva-l-learning-better-mim-representations-from-eva-clip
    'eva_large_patch14_196.in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_196px_21k_to_1k_ft_88p6.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 196, 196), crop_pct=1.0),
    'eva_large_patch14_336.in22k_ft_in22k_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_336px_21k_to_1k_ft_89p2.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),
    'eva_large_patch14_196.in22k_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_196px_1k_ft_88p0.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 196, 196), crop_pct=1.0),
    'eva_large_patch14_336.in22k_ft_in1k': _cfg(
        # hf_hub_id='BAAI/EVA', hf_hub_filename='eva_l_psz14_336px_1k_ft_88p65.pt',
        hf_hub_id='timm/', license='mit',
        mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD,
        input_size=(3, 336, 336), crop_pct=1.0, crop_mode='squash'),

    'flexivit_small.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_small.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_small.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),

    'flexivit_base.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_base.1000ep_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_1000ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),
    'flexivit_base.300ep_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),

    'flexivit_large.1200ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_large.600ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_600ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),
    'flexivit_large.300ep_in1k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95),

    'flexivit_base.patch16_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/vit_b16_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),
    'flexivit_base.patch30_in21k': _cfg(
        url='https://storage.googleapis.com/big_vision/flexivit/vit_b30_i21k_300ep.npz', custom_load=True,
        hf_hub_id='timm/',
        input_size=(3, 240, 240), crop_pct=0.95, num_classes=21843),

    'vit_base_patch16_xp_224.untrained': _cfg(url=''),
    'vit_large_patch14_xp_224.untrained': _cfg(url=''),
    'vit_huge_patch14_xp_224.untrained': _cfg(url=''),

    'vit_base_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_large_patch16_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch14_224.mae': _cfg(
        url='https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    'vit_huge_patch14_gap_224.in1k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch14_gap_224.in22k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_huge_patch16_gap_448.in1k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        input_size=(3, 448, 448), crop_pct=1.0,
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_giant_patch16_gap_224.in22k_ijepa': _cfg(
        url='https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar',
        # hf_hub_id='timm/',
        license='cc-by-nc-4.0',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),

    'vit_base_patch16_siglip_224.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0),
    'vit_base_patch16_siglip_256.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-256',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 256, 256),
        num_classes=0),
    'vit_base_patch16_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),
    'vit_base_patch16_siglip_512.webli': _cfg(
        hf_hub_id='timm/ViT-B-16-SigLIP-512',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 512, 512),
        num_classes=0),
    'vit_large_patch16_siglip_256.webli': _cfg(
        hf_hub_id='timm/ViT-L-16-SigLIP-256',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 256, 256),
        num_classes=0),
    'vit_large_patch16_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-L-16-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),
    'vit_so400m_patch14_siglip_224.webli': _cfg(
        hf_hub_id='timm/ViT-SO400M-14-SigLIP',
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=0),
    'vit_so400m_patch14_siglip_384.webli': _cfg(
        hf_hub_id='timm/ViT-SO400M-14-SigLIP-384',
        hf_hub_filename='open_clip_pytorch_model.bin',
        input_size=(3, 384, 384),
        num_classes=0),

    'vit_medium_patch16_reg4_256': _cfg(
        input_size=(3, 256, 256)),
    'vit_medium_patch16_reg4_gap_256': _cfg(
        input_size=(3, 256, 256)),
    'vit_base_patch16_reg4_gap_256': _cfg(
        input_size=(3, 256, 256)),
    'vit_so150m_patch16_reg4_gap_256': _cfg(
        input_size=(3, 256, 256)),
    'vit_so150m_patch16_reg4_map_256': _cfg(
        input_size=(3, 256, 256)),
}

_quick_gelu_cfgs = [
    'vit_large_patch14_clip_224.dfn2b',
    'vit_huge_patch14_clip_224.dfn5b',
    'vit_huge_patch14_clip_378.dfn5b',
    'vit_base_patch32_clip_224.metaclip_2pt5b',
    'vit_base_patch16_clip_224.metaclip_2pt5b',
    'vit_large_patch14_clip_224.metaclip_2pt5b',
    'vit_huge_patch14_clip_224.metaclip_2pt5b',
    'vit_base_patch32_clip_224.openai',
    'vit_base_patch16_clip_224.openai',
    'vit_large_patch14_clip_224.openai',
    'vit_large_patch14_clip_336.openai',
]
default_cfgs.update({
    n.replace('_clip_', '_clip_quickgelu_'): default_cfgs[n] for n in _quick_gelu_cfgs
})
default_cfgs = generate_default_cfgs(default_cfgs)


def _create_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> ProgResViT:
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        ProgResViT,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


def _create_progresvit(pretrained: bool = False, variant: str = 'progresvit', **kwargs) -> ProgResViT:
    """ProgResViT with width derived from the largest requested stage."""
    progress_stages = tuple(int(s) for s in kwargs.get('progress_stages', (3, 6)))
    progress_img_sizes = kwargs.get('progress_img_sizes', None)
    if progress_img_sizes is not None:
        progress_img_sizes = tuple(int(s) for s in progress_img_sizes)
    if not progress_stages:
        raise ValueError('progress_stages must not be empty')
    max_heads = max(progress_stages)
    model_args = dict(
        img_size=240,
        patch_size=16,
        embed_dim=64 * max_heads,
        depth=12,
        num_heads=max_heads,
        progress_stages=progress_stages,
        progress_img_sizes=progress_img_sizes,
    )
    model = _create_vision_transformer(variant, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def progresvit(pretrained: bool = False, **kwargs) -> ProgResViT:
    return _create_progresvit(pretrained=pretrained, variant='progresvit', **kwargs)


@register_model
def vit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_tiny_patch16_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch32_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Small (ViT-S/32)
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch32_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch16_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch8_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Small (ViT-S/8)
    """
    model_args = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch32_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch32_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch8_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch32_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch32_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch14_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/14)
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch14_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_giant_patch14_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Giant (little-g) model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_args = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_gigantic_patch14_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Gigantic (big-G) model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_args = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16)
    model = _create_vision_transformer(
        'vit_gigantic_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_224_miil(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
    model = _create_vision_transformer(
        'vit_base_patch16_224_miil', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_medium_patch16_gap_240(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 240x240
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_medium_patch16_gap_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 256x256
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_medium_patch16_gap_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 384x384
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_gap_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base (ViT-B/16) w/o class token, w/ avg-pool @ 224x224
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_base_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch14_gap_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/14) w/ no class token, avg pool
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_huge_patch14_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch16_gap_448(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/16) w/ no class token, avg pool @ 448x448
    """
    model_args = dict(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_huge_patch16_gap_448', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_giant_patch16_gap_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Giant (little-gg) model (ViT-g/16) w/ no class token, avg pool
    """
    model_args = dict(
        patch_size=16, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_giant_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch32_clip_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/32 CLIP image tower @ 224x224
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch32_clip_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/32 CLIP image tower @ 256x256
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch32_clip_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/32 CLIP image tower @ 384x384
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch32_clip_448(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/32 CLIP image tower @ 448x448
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_448', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_clip_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/16 CLIP image tower
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch16_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_clip_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/16 CLIP image tower @ 384x384
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch16_clip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch14_clip_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/14) CLIP image tower
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch14_clip_336(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/14) CLIP image tower @ 336x336
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_large_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch14_clip_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/14) CLIP image tower.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch14_clip_336(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/14) CLIP image tower @ 336x336
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch14_clip_378(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/14) CLIP image tower @ 378x378
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_378', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_giant_patch14_clip_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Giant (little-g) model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    Pretrained weights from CLIP image tower.
    """
    model_args = dict(
        patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_giant_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_gigantic_patch14_clip_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-bigG model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    Pretrained weights from CLIP image tower.
    """
    model_args = dict(
        patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_gigantic_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch32_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/32 CLIP image tower @ 224x224
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_base_patch32_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/16 CLIP image tower w/ QuickGELU act
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_base_patch16_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch14_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/14) CLIP image tower w/ QuickGELU act
    """
    from timm.layers import get_act_layer
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_large_patch14_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch14_clip_quickgelu_336(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/14) CLIP image tower @ 336x336 w/ QuickGELU act
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_large_patch14_clip_quickgelu_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch14_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/14) CLIP image tower w/ QuickGELU act.
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch14_clip_quickgelu_378(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/14) CLIP image tower @ 378x378 w/ QuickGELU act
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_quickgelu_378', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# Experimental models below

@register_model
def vit_base_patch32_plus_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base (ViT-B/32+)
    """
    model_args = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_base_patch32_plus_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_plus_240(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base (ViT-B/16+)
    """
    model_args = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_base_patch16_plus_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_rpn_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base (ViT-B/16) w/ residual post-norm
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, init_values=1e-5,
        class_token=False, block_fn=ResPostBlock, global_pool='avg')
    model = _create_vision_transformer(
        'vit_base_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch16_36x1_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base w/ LayerScale + 36 x 1 (36 block serial) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=36, num_heads=6, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_small_patch16_36x1_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch16_18x2_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Small w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=18, num_heads=6, init_values=1e-5, block_fn=ParallelThingsBlock)
    model = _create_vision_transformer(
        'vit_small_patch16_18x2_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_18x2_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Base w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=18, num_heads=12, init_values=1e-5, block_fn=ParallelThingsBlock)
    model = _create_vision_transformer(
        'vit_base_patch16_18x2_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva_large_patch14_196(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ EVA-large model https://arxiv.org/abs/2211.07636 /via MAE MIM pretrain"""
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, global_pool='avg')
    model = _create_vision_transformer(
        'eva_large_patch14_196', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva_large_patch14_336(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ EVA-large model https://arxiv.org/abs/2211.07636 via MAE MIM pretrain"""
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, global_pool='avg')
    model = _create_vision_transformer('eva_large_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def flexivit_small(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ FlexiViT-Small
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True)
    model = _create_vision_transformer('flexivit_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def flexivit_base(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ FlexiViT-Base
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True)
    model = _create_vision_transformer('flexivit_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def flexivit_large(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ FlexiViT-Large
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True)
    model = _create_vision_transformer('flexivit_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_xp_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/14) w/ parallel blocks and qk norm enabled.
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, no_embed_class=True,
        norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
    )
    model = _create_vision_transformer(
        'vit_base_patch16_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch14_xp_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Large model (ViT-L/14) w/ parallel blocks and qk norm enabled.
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, no_embed_class=True,
        norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
    )
    model = _create_vision_transformer(
        'vit_large_patch14_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_huge_patch14_xp_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-Huge model (ViT-H/14) w/ parallel blocks and qk norm enabled.
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, no_embed_class=True,
        norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
    )
    model = _create_vision_transformer(
        'vit_huge_patch14_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch14_dinov2(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-S/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'vit_small_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch14_dinov2(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'vit_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch14_dinov2(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-L/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'vit_large_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_giant_patch14_dinov2(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-G/14 for DINOv2
    """
    # The hidden_features of SwiGLU is calculated by:
    # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    # When embed_dim=1536, hidden_features=4096
    # With SwiGLUPacked, we need to set hidden_features = 2 * 4096 = 8192
    model_args = dict(
        patch_size=14, embed_dim=1536, depth=40, num_heads=24, init_values=1e-5,
        mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, img_size=518, act_layer=nn.SiLU
    )
    model = _create_vision_transformer(
        'vit_giant_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-S/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_small_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-B/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_base_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-L/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_large_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_giant_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> ProgResViT:
    """ ViT-G/14 for DINOv2
    """
    # The hidden_features of SwiGLU is calculated by:
    # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    # When embed_dim=1536, hidden_features=4096
    # With SwiGLUPacked, we need to set hidden_features = 2 * 4096 = 8192
    model_args = dict(
        patch_size=14, embed_dim=1536, depth=40, num_heads=24, init_values=1e-5, mlp_ratio=2.66667 * 2,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_giant_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_512(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_512', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_siglip_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_large_patch16_siglip_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_siglip_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_large_patch16_siglip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_so400m_patch14_siglip_224(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so400m_patch14_siglip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_so400m_patch14_siglip_384(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so400m_patch14_siglip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_medium_patch16_reg4_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=True,
        no_embed_class=True, reg_tokens=4,
    )
    model = _create_vision_transformer(
        'vit_medium_patch16_reg4_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_medium_patch16_reg4_gap_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8,
        class_token=False, no_embed_class=True, reg_tokens=4, global_pool='avg',
    )
    model = _create_vision_transformer(
        'vit_medium_patch16_reg4_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_reg4_gap_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False,
        no_embed_class=True, global_pool='avg', reg_tokens=4,
    )
    model = _create_vision_transformer(
        'vit_base_patch16_reg4_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_so150m_patch16_reg4_map_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=896, depth=18, num_heads=14, mlp_ratio=2.572,
        class_token=False, reg_tokens=4, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so150m_patch16_reg4_map_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_so150m_patch16_reg4_gap_256(pretrained: bool = False, **kwargs) -> ProgResViT:
    model_args = dict(
        patch_size=16, embed_dim=896, depth=18, num_heads=14, mlp_ratio=2.572,
        class_token=False, reg_tokens=4, global_pool='avg', fc_norm=False,
    )
    model = _create_vision_transformer(
        'vit_so150m_patch16_reg4_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


register_model_deprecations(__name__, {
    'vit_tiny_patch16_224_in21k': 'vit_tiny_patch16_224.augreg_in21k',
    'vit_small_patch32_224_in21k': 'vit_small_patch32_224.augreg_in21k',
    'vit_small_patch16_224_in21k': 'vit_small_patch16_224.augreg_in21k',
    'vit_base_patch32_224_in21k': 'vit_base_patch32_224.augreg_in21k',
    'vit_base_patch16_224_in21k': 'vit_base_patch16_224.augreg_in21k',
    'vit_base_patch8_224_in21k': 'vit_base_patch8_224.augreg_in21k',
    'vit_large_patch32_224_in21k': 'vit_large_patch32_224.orig_in21k',
    'vit_large_patch16_224_in21k': 'vit_large_patch16_224.augreg_in21k',
    'vit_huge_patch14_224_in21k': 'vit_huge_patch14_224.orig_in21k',
    'vit_base_patch32_224_sam': 'vit_base_patch32_224.sam',
    'vit_base_patch16_224_sam': 'vit_base_patch16_224.sam',
    'vit_small_patch16_224_dino': 'vit_small_patch16_224.dino',
    'vit_small_patch8_224_dino': 'vit_small_patch8_224.dino',
    'vit_base_patch16_224_dino': 'vit_base_patch16_224.dino',
    'vit_base_patch8_224_dino': 'vit_base_patch8_224.dino',
    'vit_base_patch16_224_miil_in21k': 'vit_base_patch16_224_miil.in21k',
    'vit_base_patch32_224_clip_laion2b': 'vit_base_patch32_clip_224.laion2b',
    'vit_large_patch14_224_clip_laion2b': 'vit_large_patch14_clip_224.laion2b',
    'vit_huge_patch14_224_clip_laion2b': 'vit_huge_patch14_clip_224.laion2b',
    'vit_giant_patch14_224_clip_laion2b': 'vit_giant_patch14_clip_224.laion2b',
})
