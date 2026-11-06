# AI 코딩 에이전트 다음 행동 예측 (DACON 2026)

2026 SW중심대학 디지털 경진대회 AI부문(DACON) 예선 제출 코드입니다.
대화 이력, 워크스페이스 상태, 현재 사용자 발화를 보고 AI 코딩 에이전트가 다음에 할 행동을 14개 중 하나로 분류합니다.

- 예선 최종 기록: **Macro-F1 0.79232**, T4 추론 **8분 57초** (제한 10분)
- 구성: mmBERT-base 3모델(증류 2, 일반 학습 1) int8 앙상블과 클래스별 bias
- 리더보드 점수는 어떤 계수에도 쓰지 않았습니다. 모든 계수는 학습 데이터의 OOF에서 뽑았고, 스크립트를 다시 실행하면 같은 제출물이 나옵니다.

> 대회 데이터는 재배포할 수 없어서 이 저장소에 넣지 않았습니다. DACON 대회 페이지에서 받아 `data/original_data/`에 두면 됩니다.

## 과제

| 항목 | 내용 |
|---|---|
| 입력 | 세션 메타데이터와 워크스페이스 상태, 대화와 행동 이력, 현재 사용자 발화 |
| 출력 | read_file, grep_search, list_directory, glob_pattern, edit_file, write_file, apply_patch, run_bash, run_tests, lint_or_typecheck, ask_user, plan_task, web_search, respond_only 중 하나 |
| 데이터 | train 70,000건 (9,429세션), test 30,000건 (비공개) |
| 평가 | Macro-F1, 퍼블릭 단일 리더보드 |
| 제출 | 추론 코드와 모델을 zip으로 제출. T4 16GB, 3 vCPU, 12GB RAM, 오프라인, 추론 10분 이내, zip 1GB 이하 |

## 최종 파이프라인

**1. 입력 직렬화** (`src/data.py`, 제출용은 `submit/script.py`에 같은 함수를 복제)

```
[PROMPT]  현재 사용자 발화
[STATE]   ci, git_dirty, 열린 파일 수, turn, tier, lang, codelang, pfopen, pfhist, loc, budget
[OPEN]    열린 파일 목록
[HISTORY] 전체 대화와 행동 이력 (시간순, 요약이나 압축 없음)
```

- `pfopen`: 현재 발화가 언급한 파일이 열린 파일 목록에 있는지
- `pfhist`: 현재 발화가 언급한 파일을 이전 이력에서 다뤘는지
- 토크나이저는 mmBERT-base, max_len 640 (토큰 길이 분포를 실측해서 정함)

**2. 모델 3개** (모두 `jhu-clsp/mmBERT-base`, 307M)

| 이름 | 학습 방법 | 스크립트 |
|---|---|---|
| kd_a03 | 4개 teacher의 OOF 평균 soft label로 증류. α 0.3, T 2.0, 3 epoch, lr 4e-5, seed 42 | `src/train_kd.py`, `src/train_kd_full.py` |
| kd_a03_s43 | 같은 레시피, seed 43 | `run_kd_s43.sh` |
| lr5full | hard label 전량 학습, lr 5e-5, 3 epoch | `run_lr5_full.sh` |

세 모델의 확률을 같은 비중으로 평균합니다.

**3. 클래스별 bias** (`src/fit_bias_wf1.py`)

test는 세션당 1 step만 들어 있고 train은 step 단위로 고르게 들어 있습니다. 그래서 OOF를 세션 길이의 역수로 가중한 macro-F1 기댓값을 목적함수로 두고, 클래스별 bias 14개를 결정론적으로 맞춥니다. 난수를 쓰지 않아서 같은 OOF에서는 항상 같은 bias가 나옵니다.

**4. 추론 최적화** (`src/quantize_int8.py`, `submit/script.py`)

- int8 per-channel 양자화 (모델당 308MB), GPU에서 fp16으로 복원
- 토크나이즈를 한 번만 하고 세 모델이 공유 (3 vCPU에서 반복 토크나이즈 제거)
- 길이 순으로 정렬한 배치 추론 (batch 256)
- 오프라인 로딩 (`HF_HUB_OFFLINE=1`, `local_files_only`)

## 검증 방법

- **세션 단위 5-fold** (`StratifiedGroupKFold`, seed 42 고정). 같은 세션의 step끼리 문맥을 공유해서 무작위로 나누면 누수가 생깁니다.
- **세션 균등 재평가**: OOF에서 세션당 1 step을 뽑아 test 분포에 맞춘 평가. 리더보드 점수를 예측하는 데 씁니다.
- **bias 재생성 검증**: `python -m src.verify_bias`가 제출한 bias 파일들을 OOF에서 다시 만들어 비트 단위로 같은지 확인합니다.
- **제출 전 점검**: 학습 코드와 제출 코드의 직렬화가 글자 단위로 같은지(mismatch 0), 오프라인 스모크 테스트, zip 내부 구성을 확인합니다. 로컬 처리량을 T4 시간으로 환산하는 계수는 모델 구조마다 달라서 제출로 실측했습니다(mmBERT 약 6배, EuroBERT 7.7배).

## 주요 제출 기록

| 단계 | 변경 | LB (Macro-F1) | T4 추론 |
|---|---|---|---|
| 1 | 이력을 요약해 512토큰에 맞춘 baseline | 0.6639 | |
| 2 | 이력 전체를 시간순으로 투입 | 0.7797 | |
| 3 | 4구획 직렬화, max_len 640, bias | 0.78518 | |
| 4 | 2모델 int8 앙상블 | 0.79015 | 6분 21초 |
| 13 | 증류 모델 페어, 세션 균등 기준 bias | 0.79100 | 6분 19초 |
| 20 | 3모델 앙상블, 결정론적 bias (최종) | **0.79232** | 8분 57초 |

3모델은 처음에는 T4 10분을 넘겨 실패했습니다. 토크나이즈 공유와 GPU 측 dequant를 넣은 뒤 다시 시간을 예측하고 제출해서 10분 안에 들어왔습니다.

## 재현 방법

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
# data/original_data/ 에 train.jsonl, train_labels.csv, test.jsonl 배치
```

1. teacher OOF 만들기: `src/sweep_epochs.py`, `src/sweep_save.py` (OOF는 `oof/`에 저장. 증류는 EuroBERT, fields 기본, lr 5e-5, seed 43 네 모델의 OOF 평균을 soft label로 씁니다)
2. 증류 모델: `python -m src.train_kd --alpha 0.3 --epochs 3 --tag kd_a03` 후 `python -m src.train_kd_full --alpha 0.3 --epochs 3 --out submit/model_kd03`
3. 나머지 두 모델: `bash run_kd_s43.sh`, `bash run_lr5_full.sh`
4. 양자화: `python -m src.quantize_int8 <모델 폴더> <출력.npz>`
5. bias: `python -m src.fit_bias_wf1 oof/kd_a03.npy oof/kd_a03_s43.npy oof/hp_lr5e5_ep3.npy --lim 0.8 -o submit/bias_trio_single.npy`
6. 검증: `python -m src.verify_bias`

학습은 RTX PRO 6000 Blackwell(96GB)에서 했습니다. 5-fold OOF는 모델당 수 시간, 전량 학습은 30분 남짓 걸립니다.

## 저장소 구조

```
src/                  학습, 증류, 양자화, bias, 검증 스크립트 (최종 버전)
  data.py             직렬화 단일 소스
  train_kd.py         증류 5-fold (OOF 생성)
  train_kd_full.py    증류 전량 학습
  train_full.py       hard label 전량 학습
  fit_bias_wf1.py     결정론적 클래스별 bias
  verify_bias.py      bias 비트 일치 재생성 검사
  quantize_int8.py    int8 per-channel 양자화
  sweep_*.py          하이퍼파라미터, 정규화, 손실 함수 실험
  train_llm_*.py      디코더 LLM 분류 실험 (채택 안 함)
submit/               제출용 추론 코드 (모델 가중치와 토크나이저는 미포함)
run_*.sh, run_*.out   실험 큐 스크립트와 일부 실행 로그
experiments/elise/    팀원 Elise의 병렬 실험 트랙 (아래 참고)
```

`experiments/elise/`는 팀원 Elise가 7월 4일 코드에서 갈라져 진행한 실험(FLOW 직렬화 변형, 증류 스크리닝, 시드 앙상블, 분석 도구)입니다. 루트의 최종 코드와는 독립적이라 경로와 의존성이 다를 수 있습니다. 서버 주소는 가렸습니다.

## 대회 뒤 돌아보기

대회가 끝나고 1등 팀(0.7988)의 공개 코드와 비교했습니다. 차이는 리더보드 튜닝이 아니라 입력 표현과 추론 엔지니어링이었습니다.

- 대회 중에는 사람이 판정한 오라클과 모델 사이의 차이를 되돌릴 수 없는 라벨 노이즈로 해석했습니다. 실제로는 평문 직렬화라는 입력 표현의 한계였고, 표현을 바꾸면 회수할 수 있는 부분이 있었습니다.
- 1등 팀은 결정 변수를 특수 토큰으로 바꾸고 대회 텍스트로 MLM 이어학습(TAPT)을 했습니다. 저희는 특수 토큰을 TAPT 없이 한 번 시도하고 실패하자 그 방향을 접었습니다.
- R-Drop, Label Smoothing, weighted CE, EMA는 평문 직렬화에서는 효과가 없었지만, 특수 토큰과 TAPT를 쓴 조건에서는 효과가 있었습니다.
- 추론은 1등 팀이 small 모델 8개를 base와 delta로 압축하고 early-exit, 토큰 예산 기반 동적 배칭, CPU와 GPU 병렬 처리로 5분 14초에 돌렸습니다.

여기서 얻은 원칙: 한 방향을 접기 전에 그 방향의 표준 레시피를 끝까지 채웠는지 확인하고, 실패한 실험 한 번으로 폐기하지 않는다.

## 라이선스

코드는 MIT 라이선스입니다. 대회 데이터와 사전학습 모델(mmBERT, EuroBERT 등)은 각 제공처의 약관을 따릅니다.
