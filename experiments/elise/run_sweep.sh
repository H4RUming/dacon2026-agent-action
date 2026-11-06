#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d .venv ]; then
  source .venv/bin/activate
fi

echo "=== START $(date) ==="
python -m src.sweep_epochs --max_epochs 4 --batch 64 --max_len 640 --tag fields_mmbert
echo "=== ALL DONE $(date) ==="
