# agent-behavior-prediction

DACON 2026 AI-SW 중심대학 디지털 경진대회 AI 부문: 코딩 에이전트의 다음 action을 14개 클래스 중 하나로 예측한다.

현재 목표는 단일 모델 점수를 바로 0.8로 올리는 것이 아니라, **validated fully nested teacher Macro-F1 0.8000 이상**을 먼저 만들고 그 teacher를 single FLOW/mmBERT student로 distill하는 것이다.

## Current Status

- Branch: `Elise`
- Best public LB: `0.78518`
- Best submission-compatible single OOF: `0.7765`
- Best fixed teacher seen: `0.782414`
- Current best validated fully nested teacher: `0.781780`
- Best 4090 training-method screen: `0.784865` Fold0 Macro-F1 with KL distillation
- Target: fully nested teacher `>= 0.800000`
- Remaining gap: `0.018220`

Active compute:

- PRO6000 Server: (비공개)
- RTX4090: (비공개)
- Returned node: (비공개) is no longer available. Do not schedule work there.

## Data

`data/original_data/` is present in the private repository.

| file | content |
|---|---|
| `train.jsonl` | 70,000 training samples |
| `train_labels.csv` | `id -> action` |
| `test.jsonl` | sample-format test file |
| `sample_submission.csv` | submission format |

Classes, fixed order:

```text
read_file
grep_search
list_directory
glob_pattern
edit_file
write_file
apply_patch
run_bash
run_tests
lint_or_typecheck
ask_user
plan_task
web_search
respond_only
```

Canonical split:

- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- group key: session id from `sample_id.rsplit("-step_", 1)[0]`
- fold artifact: `oof/sgkf5_split_seed42_fold_ids.npy`
- row-order artifact: `oof/train_ids.npy`

All OOF candidates must use the same row order, class order, and fold assignment.

## Core Serializers

### Fields Serializer

Source: `src/data.py`

The base input is:

```text
[PROMPT] {current_prompt}
[STATE] ci=... dirty=... open=... turn=... tier=... lang=... codelang=... loc=... budget=... pfopen=... pfhist=...
[OPEN] {open file paths}
[HISTORY]
U: ...
A: action(args...) -> result_summary
```

This remains the submission-compatible base serializer.

### FLOW Serializer

Source: `src/data_flow.py`

FLOW inserts one line after `[PROMPT]`:

```text
[FLOW] prev3=... turn=... lastres=...
```

FLOW is the strongest representation discovered so far:

- FLOW42 ep3 Macro-F1: `0.773768`
- baseline fields seed42 ep3 Macro-F1: `0.770729`
- delta: `+0.003038`

### FLOW Variants

Source: `src/data_flow_variants.py`

Currently supported variants:

- `flow_exact`: exact FLOW line
- `flow_last_action`: exact FLOW plus `last_action=...`
- `flow_prev4`: replaces `prev3` with `prev4`
- `flow_last_ci_status`: exact FLOW plus compact `last_ci_status=...`

Do not manually reimplement a promoted serializer on a remote node. Copy this source file.

## Current OOF Leaderboard

| candidate | Macro-F1 | Accuracy | artifact |
|---|---:|---:|---|
| FLOW42 ep3 | 0.773768 | 0.770557 | `oof/fields_flow_ep3.npy` |
| FLOW47 ep3 | 0.773657 | 0.770771 | `oof/fields_flow_seed47_ep3_probs.npy` |
| fields seed45 | 0.772409 | 0.770114 | `oof/fields_mmbert_seed45_probs.npy` |
| fields seed47 | 0.772105 | 0.769314 | `oof/fields_mmbert_seed47_probs.npy` |
| fields seed46 | 0.771424 | 0.769243 | `oof/fields_mmbert_seed46_probs.npy` |
| baseline fields seed42 ep3 | 0.770729 | 0.769086 | `oof/fields_mmbert_ep3.npy` |
| FLOW45 ep3 | 0.770445 | 0.769000 | `oof/fields_flow_seed45_ep3_probs.npy` |

FLOW45 is archived as weak. Do not expand that seed direction.

## Teacher State

Current best fixed greedy teacher:

```text
FLOW42
+ FLOW47        log-prob blend, weight 0.50
+ fields_seed45 log-prob blend, weight 0.35
+ fields_seed47 probability blend, weight 0.25
= Macro-F1 0.782046
= Accuracy 0.776614
```

Best fixed-only teacher seen:

```text
FLOW + last_action included
= Macro-F1 0.782414
= fixed gain +0.000368
```

This was not accepted as the validated teacher because the fully nested bias score regressed:

```text
last_action nested bias Macro-F1 0.781186
validated nested gain -0.000594
decision: REJECT_FOR_VALIDATED_TEACHER__MARGINAL_FIXED_ONLY
```

Current best fully nested teacher:

```text
raw nested:
  Macro-F1 0.781264
  Accuracy  0.776300

strict nested bias:
  Macro-F1 0.781780
  Accuracy  0.776600
```

Teacher artifacts:

- `results/search_teacher_pool_flow45_flow47.json`
- `results/search_teacher_pool_flow45_flow47.md`
- `oof/search_teacher_pool_flow45_flow47_fixed_probs.npy`
- `oof/search_teacher_pool_flow45_flow47_nested_probs.npy`
- `oof/search_teacher_pool_flow45_flow47_nested_bias_probs.npy`
- `results/search_teacher_pool_with_last_action.json`
- `results/search_teacher_pool_with_last_action.md`

Global manifests:

- `results/search_global_leaderboard.json`
- `results/search6000_fullcv_manifest.json`
- `results/search4090_screen_manifest.json`
- `results/search4090_promotions.json`

## Active Jobs

### RTX4090 Training-Method Worker

Current job:

- tmux: `search4090-distill-l050-t20-fold1`
- tag: `search4090_distill_l050_t20_fold1`
- candidate: FLOW exact student with teacher KL distillation
- source: `src/search4090_distill_screen.py`
- recipe: `lambda=0.50`, `T=2.0`, `fold=1`, `split_seed=42`, `model_seed=42`, 4 epochs, effective batch 64
- log: `logs/search4090_distill_l050_t20_fold1_20260706_101612.log`
- status as of `2026-07-06 19:28 KST`: running, RTX 4090 at about `14.2 GiB` VRAM and `99%` utilization

Completed Fold0 distillation grid:

| candidate | lambda | T | Fold0 ep3 Macro-F1 | Accuracy | Delta vs FLOW exact | decision |
|---|---:|---:|---:|---:|---:|---|
| `search4090_distill_l020_t15` | 0.20 | 1.5 | 0.777366 | 0.773373 | +0.005420 | `HIGH_PRIORITY_PROMOTION` |
| `search4090_distill_l020_t20` | 0.20 | 2.0 | 0.777608 | 0.776230 | +0.005662 | `HIGH_PRIORITY_PROMOTION` |
| `search4090_distill_l035_t15` | 0.35 | 1.5 | 0.778886 | 0.776087 | +0.006939 | `HIGH_PRIORITY_PROMOTION` |
| `search4090_distill_l035_t20` | 0.35 | 2.0 | 0.783524 | 0.779159 | +0.011578 | `HIGH_PRIORITY_PROMOTION` |
| `search4090_distill_l050_t15` | 0.50 | 1.5 | 0.784116 | 0.779801 | +0.012169 | `HIGH_PRIORITY_PROMOTION` |
| `search4090_distill_l050_t20` | 0.50 | 2.0 | 0.784865 | 0.782016 | +0.012918 | `HIGH_PRIORITY_PROMOTION` |

Interpretation:

- Distillation is the first direction here with a large Fold0 jump.
- Best Fold0 candidate is `search4090_distill_l050_t20`.
- If Fold1 confirmation remains strong, promote `lambda=0.50, T=2.0` to full 5-fold CV.
- If Fold1 regresses sharply, use `lambda=0.35, T=2.0` as the lower-distillation backup.

### PRO6000 Depth Worker

Latest completed job:

- tmux: `dacon-flow-last-action-seed42`
- tag: `fields_flow_last_action_seed42`
- candidate: `FLOW + explicit last_action`
- source: `src/data_flow_variants.py`, variant `flow_last_action`
- trainer: `src/sweep_flow_variant_seed.py`
- runner: `run_flow_last_action_seed42_pro6000.sh`
- recipe: `split_seed=42`, `model_seed=42`, 4 epochs, physical batch 64
- primary artifact: `oof/fields_flow_last_action_seed42_ep3_probs.npy`

Result:

```text
OOF ep3 Macro-F1: 0.772369
OOF ep3 Accuracy: 0.769500
fixed teacher gain: +0.000368
nested bias gain: -0.000594
decision: reject as validated teacher
```

### RTX4090 Breadth Worker

Recent completed screens:

- `search4090_flow_prev4`: `0.769857`, delta `-0.002089`, reject
- `search4090_prompt_residual`: `0.764942`, delta `-0.002299`, reject
- `search4090_inspect_aux_l020`: `0.768360`, delta `+0.001118`, Fold1 confirmation only

Do not expand these families unless a new training-method result justifies revisiting them.

Historical queue runner:

- `run_search4090_pool_aux_queue.sh`

## Experiment Rules

Do not repeat these failed directions:

- sparse/meta features
- mmBERT + sparse stacking
- binary routing
- scalar regression
- temperature-only tuning
- generic per-class scale
- epoch contextual LR
- generic family auxiliary CE
- FLOW45 seed expansion

Promotion rules for 4090 Fold0 paired screens:

| Fold0 ep3 delta | decision |
|---:|---|
| `< +0.0010` | reject |
| `+0.0010 ~ +0.0020` | Fold1 confirmation |
| `>= +0.0020` | promote full-CV |
| `>= +0.0040` | high-priority promotion |

Candidate acceptance after full-CV:

| fixed teacher gain | decision |
|---:|---|
| `>= +0.0010` | strong accept |
| `+0.0005 ~ +0.0010` | accept |
| `+0.0002 ~ +0.0005` | marginal |
| `<= +0.0002` | reject |

Nested gain must be reported separately. If nested gets worse, do not claim generalization from fixed gain alone.

## Important Scripts

Training and full-CV:

- `src/sweep_seed.py`: fixed-split fields seed full-CV
- `src/sweep_flow.py`: original FLOW42 sweep and analysis
- `src/sweep_flow_seed.py`: FLOW seed replication full-CV
- `src/sweep_flow_variant_seed.py`: FLOW variant full-CV, used for `flow_last_action`

4090 screening:

- `src/search4090_flow_screen.py`: serializer-only FLOW variant Fold0 screen
- `src/search4090_custom_screen.py`: pooling / inspection-aux Fold0 screen
- `src/search4090_distill_screen.py`: FLOW exact student with teacher KL distillation screen

Analysis:

- `tools/analyze_current_oof_ensemble.py`
- `tools/analyze_nested_greedy.py`
- `tools/analyze_search_teacher_pool.py`
- `tools/check_flow_serializer.py`

Runners:

- `run_flow_seed45_pro6000.sh`
- `run_flow_last_action_seed42_pro6000.sh`
- `run_search4090_flow_last_action.sh`
- `run_search4090_flow_prev4.sh`
- `run_search4090_prompt_residual.sh`
- `run_search4090_section_static.sh`
- `run_search4090_inspect_aux_l020.sh`
- `run_search4090_pool_aux_queue.sh`
- `run_search4090_distill_grid.sh`

## Submission Constraints

Final submission must be a single T4-compatible model:

- `submit.zip <= 1GB`
- T4 16GB
- inference within 10 minutes
- package shape: `model/`, `script.py`, `requirements.txt`

Current teacher ensemble is for search and distillation only. Do not put multi-model teacher inference into `submit/script.py`.

Submission checklist:

```bash
# serializer parity
python - <<'EOF'
import json, importlib.util
from src.data import serialize_example as A
spec = importlib.util.spec_from_file_location("S", "submit/script.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
samples = [json.loads(l) for l in open("data/original_data/test.jsonl", encoding="utf-8") if l.strip()]
print("mismatch:", sum(1 for s in samples if A(s) != S.serialize_example(s)))
EOF

# zip from submit/
cd submit
rm -rf __pycache__ output data
zip -r ../submit.zip model script.py requirements.txt -x '*__pycache__*' '*.pyc'
cd ..

# inspect zip, not filesystem
unzip -p submit.zip script.py | grep -E 'codelang|pfopen|^MAX_LEN|BIAS'
unzip -l submit.zip | grep -E 'safetensors|bias'
du -h submit.zip
```

## Repository Hygiene

Commit source and documentation only.

Ignored artifact directories:

- `oof/`
- `logs/`
- `results/`
- `checkpoints/`
- Python caches
- `submit.zip`

Do not delete artifact directories just to make the tree look clean. They are the local experiment record.
