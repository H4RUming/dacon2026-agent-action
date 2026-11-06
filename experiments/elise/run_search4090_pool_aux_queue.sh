#!/usr/bin/env bash
set -euo pipefail

cd /home/user/agent-behavior-prediction

while pgrep -f "src.search4090_flow_screen.*search4090_flow_prev4" >/dev/null; do
  sleep 300
done

bash ./run_search4090_prompt_residual.sh
bash ./run_search4090_section_static.sh
bash ./run_search4090_inspect_aux_l020.sh
