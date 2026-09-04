#!/usr/bin/env bash

# Copyright 2026 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${1:-${ROOT}/checkpoints}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON="${ROOT}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "Python was not found. Activate the project environment first." >&2
    exit 1
  fi
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


checkpoints = {
    "NCPS/progresvit-deit-s-192-240-imagenet1k": "progresvit_192_240.pth.tar",
    "NCPS/progresvit-deit-s-192-240-kd-imagenet1k": "progresvit_192_240_kd.pth.tar",
    "NCPS/progresvit-deit-s-160-384-imagenet1k": "progresvit_160_384.pth.tar",
    "NCPS/progresvit-deit-s-160-384-kd-imagenet1k": "progresvit_160_384_kd.pth.tar",
}

destination = Path(os.environ["CHECKPOINT_DIR"]).expanduser().resolve()
destination.mkdir(parents=True, exist_ok=True)
for repo_id, filename in checkpoints.items():
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=destination)
    print(path)
PY
