#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found" >&2
  exit 1
fi
conda activate DACON

mkdir -p logs results oof checkpoints
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "worker_start $(date -Is)"
echo "repo_root $ROOT"
python -V
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

for seed in 45 46 47; do
  tag="fields_mmbert_seed${seed}"
  echo "seed_start ${seed} $(date -Is)"
  python -m src.sweep_seed \
    --model jhu-clsp/mmBERT-base \
    --folds 5 \
    --split_seed 42 \
    --model_seed "$seed" \
    --epochs 3 \
    --batch 64 \
    --max_len 640 \
    --lr 4e-5 \
    --warmup 0.06 \
    --num_workers 3 \
    --tag "$tag" \
    2>&1 | tee "logs/${tag}_$(date +%Y%m%d_%H%M%S).log"
  python - "$tag" <<'PY'
import json, sys
import numpy as np
tag = sys.argv[1]
for kind in ["logits", "probs"]:
    path = f"oof/{tag}_{kind}.npy"
    arr = np.load(path)
    print(path, arr.shape, arr.dtype, float(np.nanmin(arr)), float(np.nanmax(arr)))
    assert arr.shape == (70000, 14)
    assert np.isfinite(arr).all()
with open(f"oof/{tag}_meta.json", encoding="utf-8") as f:
    meta = json.load(f)
print("meta_oof_macro_f1", meta["oof_macro_f1_probs"])
PY
  echo "seed_done ${seed} $(date -Is)"
done

python tools/analyze_seed_ensemble.py 2>&1 | tee "logs/seed_ensemble_analysis_$(date +%Y%m%d_%H%M%S).log"
echo "worker_done $(date -Is)"
