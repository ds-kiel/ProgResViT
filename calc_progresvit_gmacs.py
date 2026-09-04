#!/usr/bin/env python3

# Copyright 2026 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted in part from the Kiel University ThinkingViT codebase.

import argparse
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from timm.models import create_model


class ProgResVitMacCounter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.total_macs = 0
        self.by_type: Dict[str, int] = {}
        self.handles = []

    def reset(self) -> None:
        self.total_macs = 0
        self.by_type = {}

    def add(self, kind: str, macs: int) -> None:
        macs = int(macs)
        self.total_macs += macs
        self.by_type[kind] = self.by_type.get(kind, 0) + macs

    def _linear_hook(self, kind: str, module, args, kwargs, output) -> None:
        if not torch.is_tensor(output) or output.shape[-1] == 0:
            return
        x = args[0]
        active_in = kwargs.get('active_dim_in')
        if active_in is None:
            active_in = kwargs.get('active_dim')
        if active_in is None and len(args) > 1:
            active_in = args[1]
        if active_in is None:
            active_in = getattr(module, 'in_features', x.shape[-1])
        active_in = int(active_in)

        out_dim = int(output.shape[-1])
        positions = output.numel() // out_dim
        self.add(kind, positions * active_in * out_dim)

    def _conv2d_hook(self, kind: str, module: nn.Conv2d, args, kwargs, output) -> None:
        if not torch.is_tensor(output):
            return
        b, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        in_per_group = module.in_channels // module.groups
        self.add(kind, b * out_channels * out_h * out_w * in_per_group * kernel_h * kernel_w)

    def _attention_hook(self, module, args, kwargs, output) -> None:
        if len(args) < 3:
            return
        x, active_heads = args[0], int(args[2])
        if not torch.is_tensor(x):
            return
        batch, tokens, _ = x.shape
        self.add('attention_matmul', 2 * batch * active_heads * tokens * tokens * module.head_dim)

    def __enter__(self):
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__
            is_token_projector = name.startswith('token_projectors.')
            if class_name in ('Linear', 'QKVLinear') or isinstance(module, nn.Linear):
                kind = 'token_projector' if is_token_projector else 'linear'
                self.handles.append(module.register_forward_hook(
                    lambda mod, args, kwargs, out, kind=kind: self._linear_hook(kind, mod, args, kwargs, out),
                    with_kwargs=True,
                ))
            elif isinstance(module, nn.Conv2d):
                kind = 'token_projector' if is_token_projector else 'conv2d'
                self.handles.append(module.register_forward_hook(
                    lambda mod, args, kwargs, out, kind=kind: self._conv2d_hook(kind, mod, args, kwargs, out),
                    with_kwargs=True,
                ))
            elif class_name == 'Attention':
                self.handles.append(module.register_forward_hook(self._attention_hook, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _patch_embed_macs(model: nn.Module, batch_size: int, image_size: int, patch_size: int) -> int:
    grid = image_size // patch_size
    return (
        batch_size
        * model.embed_dim
        * grid
        * grid
        * model.patch_embed.proj.in_channels
        * patch_size
        * patch_size
    )


def _format_gmacs(macs: float) -> str:
    return f'{macs / 1e9:.4f}'


def _build_model(args, device: torch.device) -> nn.Module:
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        img_size=args.img_size,
        progress_stages=args.progress_stages,
        progress_img_sizes=args.progress_img_sizes,
    )
    model.to(device)
    model.eval()
    return model


def _run_stage(
        model: nn.Module,
        x: torch.Tensor,
        stage_idx: int,
        stages: Sequence[int],
        img_sizes: Sequence[int],
        prev_tokens=None,
):
    active_heads = int(stages[stage_idx])
    active_dim = model._stage_dim(active_heads)
    output = model.forward_features(
        x,
        active_dim,
        active_heads,
        prev_tokens,
        stage_idx=stage_idx,
        progress_img_sizes=img_sizes,
    )
    tokens = output
    logits = model.forward_head(tokens, active_dim=active_dim)
    return tokens, logits


def calculate_round_gmacs(model: nn.Module, image_size: int, batch_size: int, device: torch.device):
    stages = tuple(int(v) for v in model.progress_stages)
    img_sizes = tuple(int(v) for v in model.progress_img_sizes)
    patch_size = model.patch_embed.patch_size[0]
    x = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    results = []

    prev_tokens = None
    with torch.no_grad(), ProgResVitMacCounter(model) as counter:
        for stage_idx, (heads, stage_img_size) in enumerate(zip(stages, img_sizes)):
            counter.reset()
            counter.add('patch_embed', _patch_embed_macs(model, batch_size, stage_img_size, patch_size))
            prev_tokens, _ = _run_stage(
                model, x, stage_idx, stages, img_sizes,
                prev_tokens=prev_tokens,
            )
            round_macs = counter.total_macs / batch_size
            results.append((stage_idx, heads, patch_size, stage_img_size, round_macs, dict(counter.by_type)))

    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate GMACs for DeiT-based ProgResViT rounds.')
    parser.add_argument('--model', default='progresvit')
    parser.add_argument('--img-size', type=int, default=240)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--progress_stages', nargs='+', type=int, default=[3, 6])
    parser.add_argument('--progress_img_sizes', nargs='+', type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _build_model(args, device)
    results = calculate_round_gmacs(model, args.img_size, args.batch_size, device)

    total_macs = 0.0
    token_projector_macs = 0.0
    for round_idx, _, _, _, macs, by_type in results:
        total_macs += macs
        token_projector_macs += by_type.get('token_projector', 0) / args.batch_size
        print(f'round_{round_idx + 1}_gmacs: {_format_gmacs(macs)}')
    print(f'token_projector_gmacs: {_format_gmacs(token_projector_macs)}')
    print(f'all_gmacs: {_format_gmacs(total_macs)}')


if __name__ == '__main__':
    main()
