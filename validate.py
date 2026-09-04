#!/usr/bin/env python3

# Copyright 2026 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Uses the vendored pytorch-image-models utilities; see NOTICE.

"""Evaluate a ProgResViT checkpoint over one or more routing thresholds."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from calc_progresvit_gmacs import calculate_round_gmacs
from timm.data import create_dataset, resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model


CONFIGS = {
    "192_240": {
        "checkpoint": "progresvit_192_240.pth.tar",
        "progress_sizes": (192, 240),
        "model_img_size": 240,
        "eval_crop_size": 240,
        "init_values": None,
        "amp": False,
        "batch_size": 256,
    },
    "192_240_kd": {
        "checkpoint": "progresvit_192_240_kd.pth.tar",
        "progress_sizes": (192, 240),
        "model_img_size": 240,
        "eval_crop_size": 384,
        "init_values": 1e-6,
        "amp": False,
        "batch_size": 256,
    },
    "160_384": {
        "checkpoint": "progresvit_160_384.pth.tar",
        "progress_sizes": (160, 384),
        "model_img_size": 384,
        "eval_crop_size": 384,
        "init_values": None,
        "amp": False,
        "batch_size": 128,
    },
    "160_384_kd": {
        "checkpoint": "progresvit_160_384_kd.pth.tar",
        "progress_sizes": (160, 384),
        "model_img_size": 384,
        "eval_crop_size": 384,
        "init_values": 1e-6,
        "amp": True,
        "batch_size": 128,
    },
}

DEFAULT_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ProgResViT accuracy and GMACs over routing thresholds."
    )
    parser.add_argument("data", type=Path, help="ImageNet root or validation directory")
    parser.add_argument("--model", default="progresvit", help="model name")
    parser.add_argument("--config", choices=tuple(CONFIGS), default="160_384_kd")
    parser.add_argument("--checkpoint", type=Path,
                        help="checkpoint path (defaults to checkpoints/<config>)")
    parser.add_argument("-b", "--batch-size", type=int,
                        help="validation batch size")
    parser.add_argument("-j", "--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-interval", type=int, default=20)
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--threshold", type=float,
                                 help="evaluate one routing threshold")
    threshold_group.add_argument("--thresholds", nargs="+", type=float,
                                 default=list(DEFAULT_THRESHOLDS),
                                 help="routing thresholds to evaluate")
    return parser.parse_args()


def clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict:
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def load_checkpoint(model: torch.nn.Module, path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    state_name = ""
    if isinstance(checkpoint, dict):
        state_dict = None
        for key in ("state_dict_ema", "model_ema", "state_dict", "model"):
            if checkpoint.get(key) is not None:
                state_dict = checkpoint[key]
                state_name = key
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(clean_state_dict(state_dict), strict=True)
    return state_name or "checkpoint"


def make_loader(
        data_dir: Path,
        model: torch.nn.Module,
        crop_size: int,
        batch_size: int,
        workers: int,
        pin_memory: bool,
):
    data_config = resolve_data_config(
        {
            "img_size": crop_size,
            "input_size": None,
            "use_train_size": False,
            "mean": None,
            "std": None,
            "interpolation": "bicubic",
            "crop_pct": 0.9,
            "crop_mode": "center",
        },
        model=model,
        use_test_size=True,
        verbose=False,
    )
    transform = create_transform(
        input_size=data_config["input_size"],
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
        crop_mode=data_config["crop_mode"],
        crop_border_pixels=0,
        use_prefetcher=False,
    )
    dataset = create_dataset(
        root=str(data_dir),
        name="",
        split="validation",
        download=False,
        input_img_mode="RGB",
    )
    dataset.transform = transform
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=False,
    )
    return dataset, loader, data_config


def top10_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    top = probabilities.topk(min(10, probabilities.shape[1]), dim=1).values
    tiny = torch.finfo(top.dtype).tiny
    top = top / top.sum(dim=1, keepdim=True).clamp_min(tiny)
    return -(top * top.clamp_min(tiny).log()).sum(dim=1)


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    return logits.topk(k, dim=1).indices.eq(targets[:, None]).any(dim=1)


@torch.inference_mode()
def collect_predictions(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        progress_sizes: Sequence[int],
        amp: bool,
        log_interval: int,
):
    values = {
        key: [] for key in (
            "entropy",
            "round1_top1",
            "round2_top1",
            "round1_top5",
            "round2_top5",
        )
    }
    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp and device.type == "cuda" else nullcontext()
        )
        with amp_context:
            round1_tokens, round1_logits = model._forward_stage(
                images, 0, None, (3, 6), progress_sizes
            )
            _, round2_logits = model._forward_stage(
                images, 1, round1_tokens, (3, 6), progress_sizes
            )
        values["entropy"].append(top10_entropy(round1_logits.float()).cpu())
        values["round1_top1"].append(round1_logits.argmax(1).eq(targets).cpu())
        values["round2_top1"].append(round2_logits.argmax(1).eq(targets).cpu())
        values["round1_top5"].append(topk_correct(round1_logits, targets, 5).cpu())
        values["round2_top5"].append(topk_correct(round2_logits, targets, 5).cpu())
        if batch_index % log_interval == 0 or batch_index + 1 == len(loader):
            done = min((batch_index + 1) * loader.batch_size, len(loader.dataset))
            print(f"Evaluated {done}/{len(loader.dataset)} images", flush=True)
    return {key: torch.cat(items).numpy() for key, items in values.items()}


def threshold_rows(arrays, thresholds: Sequence[float], cumulative_gmacs):
    total = arrays["entropy"].size
    rows = []
    for threshold in thresholds:
        exit_round1 = arrays["entropy"] < threshold
        top1 = np.where(exit_round1, arrays["round1_top1"], arrays["round2_top1"])
        top5 = np.where(exit_round1, arrays["round1_top5"], arrays["round2_top5"])
        exits = int(exit_round1.sum())
        average_gmacs = (
            exits * cumulative_gmacs[0] + (total - exits) * cumulative_gmacs[1]
        ) / total
        rows.append(
            (
                threshold,
                100 * top1.mean(),
                100 * top5.mean(),
                average_gmacs,
                100 * exits / total,
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = CONFIGS[args.config]
    checkpoint = args.checkpoint or root / "checkpoints" / config["checkpoint"]
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not args.data.is_dir():
        raise SystemExit(f"ImageNet directory not found: {args.data}")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        img_size=config["model_img_size"],
        progress_stages=(3, 6),
        progress_img_sizes=config["progress_sizes"],
        init_values=config["init_values"],
    )
    state_name = load_checkpoint(model, checkpoint)
    model.to(device).eval()
    batch_size = args.batch_size or config["batch_size"]
    dataset, loader, data_config = make_loader(
        args.data,
        model,
        config["eval_crop_size"],
        batch_size,
        args.workers,
        device.type == "cuda",
    )
    if len(dataset) != 50_000:
        raise SystemExit(f"Expected 50,000 validation images, found {len(dataset)}")

    print(f"Configuration: {args.config}")
    print(f"Checkpoint: {checkpoint} ({state_name})")
    print(f"Device: {device} | Images: {len(dataset)} | Crop: {data_config['input_size'][-1]}")

    measured = calculate_round_gmacs(model, config["model_img_size"], 1, device)
    incremental = [float(item[4] / 1e9) for item in measured]
    cumulative_gmacs = np.cumsum(np.asarray(incremental)).tolist()
    arrays = collect_predictions(
        model,
        loader,
        device,
        tuple(config["progress_sizes"]),
        bool(config["amp"]),
        args.log_interval,
    )
    thresholds = [args.threshold] if args.threshold is not None else args.thresholds
    rows = threshold_rows(arrays, thresholds, cumulative_gmacs)

    print("\nRouted inference")
    print(
        f"{'Threshold':>10}  {'Top-1 (%)':>10}  {'Top-5 (%)':>10}  "
        f"{'Avg. GMACs':>11}  {'Round-1 exits (%)':>17}"
    )
    print("-" * 66)
    for threshold, top1, top5, gmacs, exit_rate in rows:
        print(
            f"{threshold:10g}  {top1:10.3f}  {top5:10.3f}  "
            f"{gmacs:11.3f}  {exit_rate:17.2f}"
        )


if __name__ == "__main__":
    main()
