# Macro-F1 개선 전략 정리

## 1. 문제 정의

본 과제는 다음 입력을 기반으로 에이전트의 다음 행동 클래스를 예측하는 14-class classification 문제다.

### 입력 구성

- `current_prompt`: 현재 사용자 발화
- `history`: 직전까지의 대화 및 에이전트 행동 이력
- `session_meta`: 요금제, 잔여 토큰 예산, 작업공간 상태, 사용 가능 도구 등 세션 메타정보

### 예측 대상

- 파일 읽기
- 파일 검색
- 파일 수정
- 셸 명령 실행
- 테스트 실행
- 사용자 질문
- 기타 탐색, 수정, 실행, 대화 행동

### 평가지표

- 14개 클래스에 대한 `Macro-F1`

`Macro-F1`은 각 클래스의 F1을 동일 가중치로 평균하므로, 빈도 높은 클래스의 accuracy보다 저빈도 클래스의 precision/recall 개선이 중요하다.

---

## 2. 핵심 개선 방향

성능 개선의 핵심은 단순히 `current_prompt`만 보는 것이 아니라, 다음 세 가지를 함께 구조화하는 것이다.

1. 현재 요청의 의도
2. 직전 행동 시퀀스
3. 작업공간 및 세션 상태

즉, 입력은 다음 형태로 정리하는 것이 유리하다.

~~~text
[CURRENT_PROMPT]
...

[RECENT_HISTORY_ACTIONS]
- user asked ...
- assistant searched file ...
- assistant read file ...
- assistant proposed patch ...
- command failed ...

[SESSION_META]
- token_budget: ...
- available_tools: ...
- open_files: ...
- dirty_files: ...
- recent_errors: ...
- workspace_state: ...

[PREDICT_NEXT_ACTION]
~~~

---

## 3. 중요 피처

## 3.1 현재 발화 기반 피처

`current_prompt`에서 다음 단서를 추출한다.

- 명령성: `해봐`, `고쳐`, `찾아`, `실행`, `테스트`, `정리`, `설명`
- 질문성: `뭐지`, `왜`, `가능?`, `맞나`, `어캐`
- 대상 객체: 파일명, 경로, 함수명, 클래스명, 에러 메시지
- 작업 유형:
  - 분석 요청
  - 검색 요청
  - 수정 요청
  - 실행 요청
  - 검증 요청
  - 설명 요청

| 표현 | 강한 후보 클래스 |
|---|---|
| `소스 분석해봐` | file_search / file_read |
| `고쳐봐` | file_modify |
| `테스트 돌려` | run_tests |
| `이 에러 뭐임` | shell_read / explain / file_search |
| `어디에 있음?` | file_search |
| `커밋해` | shell_exec |

---

## 3.2 History 기반 피처

`history`는 가장 중요한 신호다. 다음 행동은 직전 행동의 결과에 강하게 의존한다.

중요 단서:

- 직전에 파일 검색을 했는가?
  - 다음 행동은 파일 읽기일 가능성이 높음.
- 직전에 파일을 읽었는가?
  - 다음 행동은 수정 또는 추가 검색일 가능성이 높음.
- 직전에 파일을 수정했는가?
  - 다음 행동은 테스트 실행 가능성이 높음.
- 직전에 명령이 실패했는가?
  - 다음 행동은 로그 분석, 파일 수정, 재실행 가능성이 높음.
- 직전에 사용자에게 질문했는가?
  - 다음 행동은 사용자 응답 대기 또는 후속 작업 가능성이 높음.

권장 구조:

~~~text
[RECENT_ACTION_SEQUENCE]
1. search_files(query="...")
2. read_file(path="...")
3. edit_file(path="...")
4. run_shell(command="pytest ...", exit_code=1)
5. current_prompt="에러 잡아봐"
~~~

이 경우 다음 행동은 `file_modify` 또는 `file_read` 쪽으로 강하게 기운다.

---

## 3.3 Session Meta 기반 피처

`session_meta`는 텍스트 내용보다 “가능한 행동 제약”을 알려주는 신호다.

중요 항목:

- 사용 가능 도구 목록
- 파일 시스템 접근 가능 여부
- 현재 열린 파일
- 수정된 파일 여부
- 최근 실패한 명령
- 잔여 토큰 예산
- 작업공간 상태
- 제출/실행 환경 정보

특히 다음 상태는 class decision에 중요하다.

| 상태 | 의미 |
|---|---|
| `dirty_files` 있음 | 다음 행동이 테스트/검증일 가능성 증가 |
| `recent_errors` 있음 | 다음 행동이 로그 분석/수정일 가능성 증가 |
| `open_files` 있음 | 동일 파일 수정/읽기 가능성 증가 |
| token budget 낮음 | 질문/요약/직접 답변 가능성 증가 |
| tool unavailable | 도구 사용 클래스 확률 감소 |

---

## 4. Confusion Pair 분석

`Macro-F1` 개선은 전체 accuracy보다 혼동쌍을 줄이는 것이 중요하다.

우선 분석할 혼동쌍은 다음과 같다.

---

## 4.1 file_search vs file_read

검색은 위치를 모를 때, 읽기는 파일/라인/함수 위치가 특정되었을 때 선택될 가능성이 높다.

개선 단서:

- `어디`, `찾아`, `관련 파일` → `file_search`
- `열어`, `확인`, `이 파일`, `라인` → `file_read`

---

## 4.2 file_modify vs shell_exec

수정 요청인지 실행 요청인지 구분해야 한다.

개선 단서:

- `고쳐`, `패치`, `수정`, `반영` → `file_modify`
- `실행`, `돌려`, `빌드`, `설치`, `확인 명령` → `shell_exec`

---

## 4.3 run_tests vs shell_exec

테스트 실행은 일반 shell 실행과 분리해야 한다.

개선 단서:

- `pytest`, `test`, `검증`, `테스트 돌려` → `run_tests`
- `ls`, `cat`, `grep`, `pip`, `git` → `shell_exec`

---

## 4.4 ask_user vs tool_use

정보 부족으로 질문해야 하는 상황과, 도구로 바로 확인 가능한 상황을 구분해야 한다.

개선 단서:

- 경로/파일명/목표가 불명확하면 `ask_user`
- 파일, 로그, 에러 메시지가 주어져 있으면 `tool_use` 우선

---

## 5. 학습 및 평가 전략

## 5.1 Class-balanced 학습

`Macro-F1` 기준에서는 class imbalance 대응이 필수다.

후보:

- class-balanced loss
- weighted cross entropy
- focal loss
- minority class oversampling
- confusion pair hard-negative mining

---

## 5.2 Hard Negative Mining

모델이 자주 헷갈리는 pair를 별도로 뽑아 재학습한다.

우선 대상:

- `file_search` ↔ `file_read`
- `file_modify` ↔ `shell_exec`
- `run_tests` ↔ `shell_exec`
- `ask_user` ↔ `direct_answer/tool_use`

---

## 5.3 Per-class 분석

매 epoch 또는 제출 전 다음 지표를 반드시 확인한다.

- per-class precision
- per-class recall
- per-class F1
- confusion matrix
- class support
- macro-F1
- weighted-F1
- low-support class F1

`Macro-F1` 기준에서는 전체 점수보다 가장 낮은 클래스 F1을 끌어올리는 것이 중요하다.

---

## 6. 입력 포맷 권장안

기본 입력 포맷:

~~~text
You are predicting the next action of an AI coding agent.

[CURRENT_PROMPT]
{current_prompt}

[RECENT_HISTORY]
{history_summary}

[SESSION_META]
{session_meta_summary}

Available labels:
{label_list}

Predict the next action label.
~~~

`history`는 원문 전체를 무조건 넣기보다 최근 행동 중심으로 압축한다.

권장 history 요약 필드:

- 최근 사용자 요청
- 최근 assistant action type
- 최근 사용한 파일/경로
- 최근 명령어
- 최근 에러
- 현재 수정 여부
- 다음으로 남은 작업

---

## 7. 구현 우선순위

## 7.1 1단계: Baseline 확인

- 기존 학습 코드 실행
- class distribution 확인
- per-class F1 출력
- confusion matrix 저장

---

## 7.2 2단계: 입력 템플릿 개선

- `current_prompt` 단독 입력과 비교
- `current_prompt + history` 비교
- `current_prompt + history + session_meta` 비교

---

## 7.3 3단계: 클래스 불균형 대응

- weighted cross entropy
- focal loss
- oversampling
- minority class augmentation

---

## 7.4 4단계: 혼동쌍 보강

- search/read
- modify/shell
- test/shell
- ask/tool

---

## 7.5 5단계: 제출 안정화

- seed 고정
- validation split 고정
- label mapping 고정
- inference output format 검증
- `submit/requirements.txt` 최소 유지

---

## 8. 주의사항

제출 환경에서 기본 제공되는 주요 패키지는 불필요하게 `requirements.txt`에 넣지 않는다.

특히 `submit/requirements.txt`에는 다음 패키지를 추가하지 않는다.

- torch
- torchvision
- torchaudio
- numpy
- pandas
- scikit-learn
- accelerate
- sentencepiece

현재 제출 요구사항은 다음 한 줄 유지가 안전하다.

~~~text
transformers==4.49.0
~~~

---

## 9. 최종 목표

목표는 단순 정확도 향상이 아니라 14개 클래스 전체의 균형 성능 향상이다.

최적화 기준은 다음 순서로 둔다.

1. Macro-F1
2. 저성능 클래스 F1
3. 주요 confusion pair 감소
4. validation 안정성
5. inference 제출 형식 안정성
