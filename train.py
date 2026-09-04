#!/usr/bin/env python3

""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""

# Copyright 2026 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Based on pytorch-image-models (timm); see NOTICE.
#
# Modifications:
# - Added ProgResViT progressive-stage training and knowledge distillation.

import warnings
warnings.filterwarnings('ignore')
from absl import logging
logging.set_verbosity(logging.ERROR)

import argparse
import importlib
import json
import logging
import math
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from timm.utils.model import _freeze_unfreeze

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm, resample_abs_pos_embed, resample_patch_embed
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

# The first parser resolves a built-in ProgResViT preset or a YAML configuration file.
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='CONFIG',
                    help='training preset or YAML config file')

TRAINING_PRESETS = {
    '192_240': {
        'model': 'progresvit',
        'experiment': 'progresvit-192-240',
        'img_size': 240,
        'input_size': [3, 240, 240],
        'progress_stages': [3, 6],
        'progress_img_sizes': [192, 240],
        'kd': False,
    },
    '192_240_kd': {
        'model': 'progresvit',
        'experiment': 'progresvit-192-240-kd',
        'img_size': 240,
        'input_size': [3, 384, 384],
        'progress_stages': [3, 6],
        'progress_img_sizes': [192, 240],
        'kd': True,
    },
    '160_384': {
        'model': 'progresvit',
        'experiment': 'progresvit-160-384',
        'img_size': 384,
        'input_size': [3, 384, 384],
        'progress_stages': [3, 6],
        'progress_img_sizes': [160, 384],
        'kd': False,
    },
    '160_384_kd': {
        'model': 'progresvit',
        'experiment': 'progresvit-160-384-kd',
        'img_size': 384,
        'input_size': [3, 384, 384],
        'progress_stages': [3, 6],
        'progress_img_sizes': [160, 384],
        'kd': True,
    },
}


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', '--data', dest='data_dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
parser.add_argument('--train-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in train split, for IterableDatasets.')
parser.add_argument('--val-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in validation split, for IterableDatasets.')
group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')
group.add_argument('--input-img-mode', default=None, type=str,
                   help='Dataset image conversion mode for input images.')
group.add_argument('--input-key', default=None, type=str,
                   help='Dataset key for input images.')
group.add_argument('--target-key', default=None, type=str,
                   help='Dataset key for target labels.')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='progresvit', type=str, metavar='MODEL',
                   help='Name of model to train (default: "progresvit")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--pretrained-path', default=None, type=str,
                   help='Load this checkpoint as if they were the pretrained weights (with adaptation).')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization (default: none)')
group.add_argument('--initial-model', default='', type=str, metavar='MODEL',
                   help='Public pretrained model used to initialize ProgResViT (downloaded automatically)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                   metavar='N N N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                   help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, action='store_true',
                   help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
group.add_argument('--head-init-scale', default=None, type=float,
                   help='Head initialization scale')
group.add_argument('--head-init-bias', default=None, type=float,
                   help='Head initialization bias value')

# Knowledge distillation
group = parser.add_argument_group('Knowledge distillation parameters')
group.add_argument('--distillation-type', default='none', choices=('none', 'soft', 'hard'),
                   help='Logit-level teacher distillation type (default: none)')
group.add_argument('--distillation-alpha', default=0.5, type=float,
                   help='Weight assigned to distillation loss; supervised loss uses 1-alpha (default: 0.5)')
group.add_argument('--distillation-tau', default=2.0, type=float,
                   help='Temperature for soft logit distillation (default: 2.0)')
group.add_argument('--teacher-model', default='', type=str,
                   help='Frozen teacher model name (required when distillation is enabled)')
group.add_argument('--teacher-pretrained', action='store_true', default=False,
                   help='Load pretrained weights for the teacher model')
group.add_argument('--teacher-checkpoint', default='', type=str, metavar='PATH',
                   help='Optional teacher checkpoint; loaded after teacher model creation')
group.add_argument('--kd', action='store_true', default=False,
                   help='Use the hard-KD recipe of the released ProgResViT checkpoints')

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

# Device & distributed
group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
group.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--dist-backend', default=None, type=str,
                   help='Distributed backend override (for example, nccl or gloo).')
group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
group.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--device-modules', default=None, type=str, nargs='+',
                    help="Python imports for device backend modules.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
group.add_argument('--train-crop-mode', type=str, default=None,
                   help='Crop-mode in train'),
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
group.add_argument('--color-jitter-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying any color jitter.')
group.add_argument('--grayscale-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying random grayscale conversion.')
group.add_argument('--gaussian-blur-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying gaussian blur.')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-sum', action='store_true', default=False,
                   help='Sum over classes when using BCE loss.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled).')
group.add_argument('--bce-pos-weight', type=float, default=None,
                   help='Positive weighting for BCE loss.')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                   help='Enable tracking moving average of model weights.')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='Decay factor for model weights moving average (default: 0.9998)')
group.add_argument('--model-ema-warmup', action='store_true',
                   help='Enable warmup for model EMA decay.')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--save-images', action='store_true', default=False,
                   help='save images of input bathes every log interval for debugging')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
group.add_argument('--eval-every', default=1, type=int, metavar='N',
                   help='Run validation every N epochs, and always on the final epoch (default: 1)')
group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
parser.add_argument('--progress_stages', nargs='+', type=int, default=[3, 6], help='progressive head counts')
parser.add_argument('--progress_img_sizes', nargs='+', type=int, default=None,
                    help='per-progress-stage input resolutions, e.g. --progress_img_sizes 192 240')


def _parse_args():
    # Resolve a built-in preset or a user-supplied YAML configuration.
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        config_name = args_config.config
        if config_name in TRAINING_PRESETS:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'args.yaml')
        else:
            config_path = config_name
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        if config_name in TRAINING_PRESETS:
            cfg.update(TRAINING_PRESETS[config_name])
        parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def _normalize_progresvit_model_name(args):
    if args.model.startswith('deit_'):
        _logger.info('Using progresvit instead of configured DeiT model %s.', args.model)
        args.model = 'progresvit'


def _apply_progresvit_training_recipe(args):
    """Resolve the public initializer and optional released KD recipe."""
    if args.initial_checkpoint and args.initial_model:
        raise ValueError('Use --initial-checkpoint or --initial-model, not both.')

    if args.model != 'progresvit':
        if args.kd:
            raise ValueError('--kd is only supported with --model progresvit.')
        return

    if args.kd:
        args.distillation_type = 'hard'
        args.distillation_alpha = 0.5
        args.teacher_model = 'deit3_base_patch16_384.fb_in22k_ft_in1k'
        args.teacher_pretrained = True
        args.amp = True
        args.amp_dtype = 'bfloat16'
        args.model_kwargs = dict(args.model_kwargs)
        args.model_kwargs.setdefault('init_values', 1e-6)
        if not args.initial_checkpoint and not args.initial_model:
            args.initial_model = 'deit3_small_patch16_384.fb_in22k_ft_in1k'
    elif not args.initial_checkpoint and not args.initial_model:
        args.initial_model = 'deit_small_patch16_224.fb_in1k'


def _load_torch_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path, map_location='cpu', weights_only=True)


def _clean_state_dict_keys(state_dict):
    return OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())


def _extract_initial_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('state_dict_ema', 'model_ema', 'state_dict', 'model'):
            if key in checkpoint:
                return _clean_state_dict_keys(checkpoint[key])
    return _clean_state_dict_keys(checkpoint)


def _crop_initial_tensor(name, value, target, model):
    if not torch.is_tensor(value) or value.ndim != target.ndim:
        return None

    if name == 'pos_embed':
        if value.shape[-1] < target.shape[-1]:
            return None
        value = value[..., :target.shape[-1]]
        if value.shape[1] != target.shape[1]:
            target_grid_size = tuple(int(size) for size in model.patch_embed.grid_size)
            target_grid_tokens = math.prod(target_grid_size)
            target_prefix_tokens = target.shape[1] - target_grid_tokens
            preferred_prefix_tokens = (
                0 if getattr(model, 'no_embed_class', False)
                else getattr(model, 'num_prefix_tokens', 1)
            )
            source_prefix_tokens = None
            for candidate in dict.fromkeys((preferred_prefix_tokens, 0, 1, 2)):
                grid_tokens = value.shape[1] - candidate
                grid_side = math.isqrt(grid_tokens) if grid_tokens >= 0 else -1
                if grid_side * grid_side == grid_tokens:
                    source_prefix_tokens = candidate
                    break
            if source_prefix_tokens is None or target_prefix_tokens < 0:
                return None
            value = resample_abs_pos_embed(
                value,
                new_size=target_grid_size,
                num_prefix_tokens=source_prefix_tokens,
            )
            if source_prefix_tokens != target_prefix_tokens:
                source_prefix = value[:, :source_prefix_tokens]
                grid = value[:, source_prefix_tokens:]
                if source_prefix_tokens >= target_prefix_tokens:
                    target_prefix = source_prefix[:, :target_prefix_tokens]
                else:
                    padding = value.new_zeros(
                        value.shape[0],
                        target_prefix_tokens - source_prefix_tokens,
                        value.shape[2],
                    )
                    target_prefix = torch.cat((source_prefix, padding), dim=1)
                value = torch.cat((target_prefix, grid), dim=1)
        return value.contiguous() if value.shape == target.shape else None

    if name == 'patch_embed.proj.weight':
        if value.shape[0] < target.shape[0] or value.shape[1] < target.shape[1]:
            return None
        value = value[:target.shape[0], :target.shape[1]]
        if value.shape[-2:] != target.shape[-2:]:
            value = resample_patch_embed(value, target.shape[-2:])
        return value.contiguous() if value.shape == target.shape else None

    if any(source < dest for source, dest in zip(value.shape, target.shape)):
        return None

    if '.attn.qkv.' in name and value.shape[0] % 3 == 0 and target.shape[0] % 3 == 0:
        source_dim = value.shape[0] // 3
        target_dim = target.shape[0] // 3
        chunks = [value[i * source_dim:i * source_dim + target_dim] for i in range(3)]
        value = torch.cat(chunks, dim=0)
        if value.ndim == 2:
            value = value[:, :target.shape[1]]
        return value.contiguous()

    slices = tuple(slice(0, dim) for dim in target.shape)
    return value[slices].contiguous()


def _adapt_initial_checkpoint(state_dict, model):
    model_state = model.state_dict()
    adapted = OrderedDict()
    cropped, skipped = [], []

    for name, value in state_dict.items():
        target = model_state.get(name)
        if target is None:
            continue
        if torch.is_tensor(value) and value.shape == target.shape:
            adapted[name] = value
            continue
        cropped_value = _crop_initial_tensor(name, value, target, model)
        if cropped_value is not None and cropped_value.shape == target.shape:
            adapted[name] = cropped_value
            cropped.append(name)
        else:
            skipped.append(name)

    if cropped:
        _logger.info('Cropped %d initial-checkpoint tensors to match progresvit width.', len(cropped))
    if skipped:
        _logger.warning('Skipped %d incompatible initial-checkpoint tensors.', len(skipped))
    return adapted


def load_initial_checkpoint(model, checkpoint_path):
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    state_dict = _extract_initial_state_dict(checkpoint)
    state_dict = _adapt_initial_checkpoint(state_dict, model)
    return model.load_state_dict(state_dict, strict=False)


def load_initial_model(model, model_name):
    _logger.info('Loading public initializer %s.', model_name)
    source_model = create_model(model_name, pretrained=True)
    state_dict = _adapt_initial_checkpoint(source_model.state_dict(), model)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    del source_model
    return incompatible_keys


def _validate_distillation_args(args):
    if args.distillation_type == 'none':
        return
    if not args.teacher_model:
        raise ValueError('--teacher-model is required when distillation is enabled.')
    if not args.teacher_pretrained and not args.teacher_checkpoint:
        raise ValueError(
            'The teacher would be randomly initialized. Use --teacher-pretrained '
            'or provide --teacher-checkpoint.'
        )
    if not 0.0 <= args.distillation_alpha <= 1.0:
        raise ValueError('--distillation-alpha must be between 0 and 1.')
    if args.distillation_tau <= 0.0:
        raise ValueError('--distillation-tau must be greater than zero.')


def _create_teacher_model(args, device, student_data_config):
    if args.distillation_type == 'none':
        return None

    teacher_model = create_model(
        args.teacher_model,
        pretrained=args.teacher_pretrained,
    )
    if args.teacher_checkpoint:
        load_checkpoint(teacher_model, args.teacher_checkpoint)

    teacher_num_classes = getattr(teacher_model, 'num_classes', None)
    if teacher_num_classes != args.num_classes:
        raise ValueError(
            f'Teacher class count ({teacher_num_classes}) does not match '
            f'student class count ({args.num_classes}).'
        )

    teacher_data_config = resolve_data_config({}, model=teacher_model)
    for field in ('input_size', 'mean', 'std'):
        teacher_value = tuple(teacher_data_config[field])
        student_value = tuple(student_data_config[field])
        if teacher_value != student_value:
            raise ValueError(
                f'Teacher and student {field} must match for shared-input distillation: '
                f'{teacher_value} vs {student_value}.'
            )

    teacher_model.to(device=device)
    if args.channels_last:
        teacher_model.to(memory_format=torch.channels_last)
    teacher_model.requires_grad_(False)
    teacher_model.eval()

    if utils.is_primary(args):
        _logger.info(
            'Using frozen teacher %s for %s last-layer distillation '
            '(alpha=%.3f, tau=%.3f, input=%s).',
            args.teacher_model,
            args.distillation_type,
            args.distillation_alpha,
            args.distillation_tau,
            teacher_data_config['input_size'],
        )
    return teacher_model


def _forward_teacher_logits(teacher_model, input):
    # This repository aliases the standard timm VisionTransformer class to
    # ProgResViT. Models created without progressive stages (including DeiT-III)
    # therefore need an explicit full-width single-pass forward.
    progress_stages = getattr(teacher_model, 'progress_stages', None)
    if progress_stages is not None and len(progress_stages) == 0:
        active_dim = int(teacher_model.embed_dim)
        active_heads = int(teacher_model.num_heads)
        features = teacher_model.forward_features(
            input,
            active_dim=active_dim,
            active_heads=active_heads,
            stage_idx=0,
            progress_img_sizes=(int(input.shape[-1]),),
        )
        return teacher_model.forward_head(features, active_dim=active_dim)
    return teacher_model(input)


def _last_layer_distillation_loss(student_logits, teacher_logits, distillation_type, tau):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f'Student and teacher logits must have identical shapes, got '
            f'{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}.'
        )
    if distillation_type == 'soft':
        student_log_probs = F.log_softmax(student_logits.float() / tau, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach().float() / tau, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (tau ** 2)
    if distillation_type == 'hard':
        return F.cross_entropy(student_logits, teacher_logits.detach().argmax(dim=-1))
    raise ValueError(f'Unsupported distillation type: {distillation_type}')


def _model_training_config_text(model, args):
    lines = [
        'Model training config:',
        f'  model: {args.model}',
        f'  num_classes: {getattr(model, "num_classes", None)}',
        f'  embed_dim: {getattr(model, "embed_dim", None)}',
        f'  num_heads: {getattr(model, "num_heads", None)}',
        f'  head_dim: {getattr(model, "head_dim", None)}',
        f'  depth: {len(getattr(model, "blocks", []))}',
    ]

    patch_embed = getattr(model, 'patch_embed', None)
    if patch_embed is not None:
        lines.extend([
            f'  img_size: {getattr(patch_embed, "img_size", None)}',
            f'  base_patch_size: {getattr(patch_embed, "patch_size", None)}',
            f'  base_grid_size: {getattr(patch_embed, "grid_size", None)}',
        ])

    progress_stages = tuple(getattr(model, 'progress_stages', ()) or ())
    img_sizes = tuple(getattr(model, 'progress_img_sizes', ()) or ())
    patch_size = getattr(model, 'progress_patch_size', None)
    if patch_size is None and patch_embed is not None:
        patch_size = getattr(patch_embed, 'patch_size', None)
    if progress_stages:
        lines.extend([
            f'  progress_stages: {progress_stages}',
            f'  progress_patch_size: {patch_size}',
            f'  progress_img_sizes: {img_sizes}',
            f'  loop_condition_dim: {getattr(model, "loop_cond_dim", None)}',
            '  loop_condition: [round_id, global_progress, cur_res_log2, prev_res_log2, delta_res_log2]',
        ])
        for idx, (heads, img_size) in enumerate(zip(progress_stages, img_sizes)):
            grid = (
                model._grid_size_for_img_size(img_size, patch_size)
                if hasattr(model, '_grid_size_for_img_size') else None
            )
            tokens = grid[0] * grid[1] if grid is not None else None
            active_dim = model._stage_dim(heads) if hasattr(model, '_stage_dim') else None
            lines.append(
                f'  round {idx + 1}: heads={heads}, img_size={img_size}, patch={patch_size}, '
                f'grid={grid}, tokens={tokens}, active_dim={active_dim}'
            )

        token_projectors = getattr(model, 'token_projectors', [])
        for idx, projector in enumerate(token_projectors):
            depthwise = getattr(projector, 'depthwise_proj', None)
            depthwise_text = (
                f', depthwise_kernel={getattr(depthwise, "kernel_size", None)}, groups={getattr(depthwise, "groups", None)}'
                if depthwise is not None else ''
            )
            lines.append(
                f'  token_projector {idx + 1}: '
                f'{getattr(projector, "source_hw", None)}->{getattr(projector, "target_hw", None)}, '
                f'channels={getattr(projector, "in_channels", None)}->{getattr(projector, "out_channels", None)}'
                f'{depthwise_text}'
            )
        fusion_input_delta = getattr(model, 'fusion_input_delta', [])
        fusion_prev_delta = getattr(model, 'fusion_prev_delta', [])
        for idx, (input_head, prev_head) in enumerate(zip(fusion_input_delta, fusion_prev_delta)):
            lines.append(
                f'  fusion_condition_heads {idx + 1}: '
                f'input_delta={input_head.__class__.__name__}, prev_delta={prev_head.__class__.__name__}, '
                f'final_layer_zero_init=True'
            )
        if getattr(model, 'loop_cond_dim', None) is not None:
            lines.append('  block_condition_gates: independent gate_attn and gate_mlp per transformer block')

    return '\n'.join(lines)


def _skipped_eval_metrics() -> OrderedDict:
    return OrderedDict([('loss', None), ('top1', None), ('top5', None)])


def _save_last_checkpoint_only(saver, epoch):
    tmp_save_path = os.path.join(saver.checkpoint_dir, 'tmp' + saver.extension)
    last_save_path = os.path.join(saver.checkpoint_dir, 'last' + saver.extension)
    saver._save(tmp_save_path, epoch)
    if os.path.exists(last_save_path):
        os.unlink(last_save_path)
    os.rename(tmp_save_path, last_save_path)


def _progress_eval_stage_indices(progress_stages):
    # Validate largest stage first so checkpoint selection keeps using the
    # strongest full-capacity metric, while still logging smaller stages.
    return tuple(reversed(range(len(progress_stages))))


def _progress_eval_log_suffix(progress_stages, stage_idx, prefix=''):
    return f'{prefix}-{int(progress_stages[stage_idx])} HEADS'


def _clear_pos_embed_eval_caches(model):
    """Drop resized positional-embedding caches before live validation.

    EMA updates use foreach tensor ops that do not bump Tensor._version, so a
    cache keyed by pos_embed._version can become stale while EMA weights change.
    Clearing here keeps training-time validation equivalent to fresh checkpoint
    evaluation.
    """
    cleared = 0
    seen = set()
    stack = [model]

    while stack:
        module = stack.pop()
        if not isinstance(module, nn.Module):
            continue

        for submodule in module.modules():
            module_id = id(submodule)
            if module_id in seen:
                continue
            seen.add(module_id)

            cache = getattr(submodule, '_pos_embed_eval_cache', None)
            if cache is not None:
                cleared += len(cache)
                cache.clear()

            # Be explicit for common wrappers such as DDP, ModelEmaV3, and
            # torch.compile OptimizedModule.
            for attr_name in ('module', '_orig_mod'):
                wrapped = getattr(submodule, attr_name, None)
                if isinstance(wrapped, nn.Module) and id(wrapped) not in seen:
                    stack.append(wrapped)

    return cleared


def _validate_progress_stages(
        model,
        model_ema,
        loader_eval,
        loss_fn,
        args,
        device,
        amp_autocast,
        log_prefix='',
):
    eval_metrics = None
    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
        if utils.is_primary(args):
            _logger.info("Distributing BatchNorm running means and vars")
        utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

    for stage_idx in _progress_eval_stage_indices(args.progress_stages):
        stage_metrics = validate(
            model,
            loader_eval,
            loss_fn,
            args,
            device=device,
            amp_autocast=amp_autocast,
            log_suffix=_progress_eval_log_suffix(args.progress_stages, stage_idx, prefix=log_prefix),
            progress_stage_idx=stage_idx,
        )
        if stage_idx == len(args.progress_stages) - 1:
            eval_metrics = stage_metrics

    if model_ema is not None and not args.model_ema_force_cpu:
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
        for stage_idx in _progress_eval_stage_indices(args.progress_stages):
            stage_metrics = validate(
                model_ema,
                loader_eval,
                loss_fn,
                args,
                device=device,
                amp_autocast=amp_autocast,
                log_suffix=_progress_eval_log_suffix(args.progress_stages, stage_idx, prefix=f'{log_prefix} (EMA)'),
                progress_stage_idx=stage_idx,
            )
            if stage_idx == len(args.progress_stages) - 1:
                eval_metrics = stage_metrics

    return eval_metrics


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()
    _normalize_progresvit_model_name(args)
    _apply_progresvit_training_recipe(args)
    args.eval_every = max(1, args.eval_every)
    _validate_distillation_args(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    print(args)
    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model_size_kwargs = {}
    if args.img_size is not None:
        model_size_kwargs['img_size'] = args.img_size
    elif args.input_size is not None:
        model_size_kwargs['img_size'] = tuple(args.input_size[-2:])

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        progress_stages=args.progress_stages,
        progress_img_sizes=args.progress_img_sizes,
        **model_size_kwargs,
        **factory_kwargs,
        **args.model_kwargs,
    )
    if args.initial_checkpoint:
        load_initial_checkpoint(model, args.initial_checkpoint)
    elif args.initial_model:
        load_initial_model(model, args.initial_model)

    if utils.is_primary(args):
        print(_model_training_config_text(model, args))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"After freezing: Total number of trainable parameters: {total_params}\n")

    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    teacher_model = _create_teacher_model(args, device, data_config)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        try:
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device='cpu' if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb, find_unused_parameters=False)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile)

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
    )

    if args.val_split:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            input_img_mode=input_img_mode,
            input_key=args.input_key,
            target_key=args.target_key,
            num_samples=args.val_num_samples,
        )

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None
    if args.val_split:
        eval_workers = args.workers
        if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, args.workers)
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=eval_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            device=device,
            use_prefetcher=args.prefetcher,
        )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric if loader_eval is not None else 'loss'
    decreasing_metric = eval_metric == 'loss'
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    results = []
    summary_path = os.path.join(output_dir, 'summary.csv') if output_dir is not None else None
    summary_header_written = bool(summary_path and os.path.exists(summary_path) and os.path.getsize(summary_path) > 0)

    if loader_eval is not None:
        if utils.is_primary(args):
            _logger.info('Running initial validation before training starts at epoch %d.', start_epoch)
        _validate_progress_stages(
            model,
            model_ema,
            loader_eval,
            validate_loss_fn,
            args,
            device=device,
            amp_autocast=amp_autocast,
            log_prefix=' (initial)',
        )

    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)
            eval_this_epoch = (
                loader_eval is not None and
                ((epoch + 1) % args.eval_every == 0 or epoch == num_epochs - 1)
            )
            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                teacher_model=teacher_model,
                mixup_fn=mixup_fn,
                num_updates_total=num_epochs * updates_per_epoch,
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if eval_this_epoch:
                eval_metrics = None
                for stage_idx in _progress_eval_stage_indices(args.progress_stages):
                    stage_metrics = validate(
                        model,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        log_suffix=_progress_eval_log_suffix(args.progress_stages, stage_idx),
                        progress_stage_idx=stage_idx,
                    )
                    if stage_idx == len(args.progress_stages) - 1:
                        eval_metrics = stage_metrics

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                    for stage_idx in _progress_eval_stage_indices(args.progress_stages):
                        stage_metrics = validate(
                            model_ema,
                            loader_eval,
                            validate_loss_fn,
                            args,
                            device=device,
                            amp_autocast=amp_autocast,
                            log_suffix=_progress_eval_log_suffix(args.progress_stages, stage_idx, prefix=' (EMA)'),
                            progress_stage_idx=stage_idx,
                        )
                        if stage_idx == len(args.progress_stages) - 1:
                            eval_metrics = stage_metrics

            else:
                eval_metrics = None

            summary_eval_metrics = eval_metrics
            if loader_eval is not None and summary_eval_metrics is None:
                summary_eval_metrics = _skipped_eval_metrics()

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    summary_eval_metrics,
                    filename=summary_path,
                    lr=sum(lrs) / len(lrs),
                    write_header=not summary_header_written,
                    log_wandb=args.log_wandb and has_wandb,
                )
                summary_header_written = True

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            elif eval_metric in train_metrics:
                latest_metric = train_metrics[eval_metric]
            else:
                latest_metric = None

            if saver is not None and latest_metric is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)
            elif saver is not None:
                _save_last_checkpoint_only(saver, epoch)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, latest_metric)

            results.append({
                'epoch': epoch,
                'train': train_metrics,
                'validation': summary_eval_metrics,
            })

    except KeyboardInterrupt:
        pass

    results = {'all': results}
    if best_metric is not None:
        results['best'] = results['all'][best_epoch - start_epoch]
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    print(f'--result\n{json.dumps(results, indent=4)}')

def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
        teacher_model=None,
        mixup_fn=None,
        num_updates_total=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    supervised_losses_m = utils.AverageMeter() if teacher_model is not None else None
    distillation_losses_m = utils.AverageMeter() if teacher_model is not None else None

    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                teacher_logits = None
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_logits = _forward_teacher_logits(teacher_model, input)
                    if isinstance(teacher_logits, (tuple, list)):
                        teacher_logits = teacher_logits[0]
                    if not torch.is_tensor(teacher_logits):
                        raise TypeError(
                            f'Teacher must return logits as a tensor, got {type(teacher_logits).__name__}.'
                        )

                outputs = model(
                    input, threshold=None, progress_stages=args.progress_stages,
                    progress_img_sizes=args.progress_img_sizes)
                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)
                # Equal supervision for each progressive stage; this keeps
                # training independent of whether the schedule has 2 or 3 stages.
                supervised_loss = torch.stack([
                    loss_fn(output, target) for output in outputs
                ]).mean()

                distillation_loss = None
                if teacher_logits is not None:
                    distillation_loss = torch.stack([
                        _last_layer_distillation_loss(
                            output,
                            teacher_logits,
                            args.distillation_type,
                            args.distillation_tau,
                        )
                        for output in outputs
                    ]).mean()
                    loss = (
                        (1.0 - args.distillation_alpha) * supervised_loss +
                        args.distillation_alpha * distillation_loss
                    )
                else:
                    loss = supervised_loss

            if accum_steps > 1:
                loss /= accum_steps
            return loss, supervised_loss.detach(), (
                distillation_loss.detach() if distillation_loss is not None else None
            )

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss, supervised_loss, distillation_loss = _forward()
                _backward(loss)
        else:
            loss, supervised_loss, distillation_loss = _forward()
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
            if supervised_losses_m is not None:
                supervised_losses_m.update(supervised_loss.item(), input.size(0))
                distillation_losses_m.update(distillation_loss.item(), input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                if supervised_losses_m is not None:
                    reduced_supervised_loss = utils.reduce_tensor(supervised_loss.data, args.world_size)
                    reduced_distillation_loss = utils.reduce_tensor(distillation_loss.data, args.world_size)
                    supervised_losses_m.update(reduced_supervised_loss.item(), input.size(0))
                    distillation_losses_m.update(reduced_distillation_loss.item(), input.size(0))
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                progress_denom = max(1, updates_per_epoch - 1)
                loss_text = f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                if supervised_losses_m is not None:
                    loss_text += (
                        f'CE: {supervised_losses_m.val:#.3g} ({supervised_losses_m.avg:#.3g})  '
                        f'KD: {distillation_losses_m.val:#.3g} ({distillation_losses_m.avg:#.3g})  '
                    )
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * update_idx / progress_denom:>3.0f}%)]  '
                    f'{loss_text}'
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    metrics = OrderedDict([('loss', losses_m.avg)])
    if supervised_losses_m is not None:
        metrics['ce_loss'] = supervised_losses_m.avg
        metrics['kd_loss'] = distillation_losses_m.avg
    return metrics


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix='',
        progress_stage_idx=None,
):
    _clear_pos_embed_eval_caches(model)
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()
    if progress_stage_idx is None:
        progress_stage_idx = len(args.progress_stages) - 1

    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(
                    input, threshold=None, progress_stages=args.progress_stages,
                    progress_img_sizes=args.progress_img_sizes,
                    eval_stage_idx=progress_stage_idx)

                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
