#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

python tools/check_submit.py

cd submit
rm -rf __pycache__ output data
zip -r ../submit.zip model script.py requirements.txt -x '*__pycache__*' '*.pyc'
cd ..

unzip -l submit.zip | sed -n '1,80p'
du -h submit.zip
