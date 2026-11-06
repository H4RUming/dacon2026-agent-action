"""
데이터 로딩 + 직렬화.
학습(train)과 제출(script.py)이 반드시 동일한 serialize_example()을 씀.
전처리 드리프트는 조용한 성능 저하나 마찬가지일 것으로 예상.
"""
import json
import os
from typing import Any

# 14개 클래스 (대소문자 고정. 제출 시 정확히 이 문자열 일치해야함.)
CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}
ID2CLASS = {i: c for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def session_id(sample_id: str) -> str:
    """'sess_..._024730-step_08' -> 'sess_..._024730' (GroupKFold 그룹 키)."""
    return sample_id.rsplit("-step_", 1)[0]


def _fmt_action(h: dict) -> str:
    """history의 assistant_action 한 개를 압축 문자열로."""
    name = h.get("name", "")
    args = h.get("args", {}) or {}
    # args는 핵심만 기록. (path/pattern/target/n_files 등) 길지 않게.
    arg_str = " ".join(f"{k}={v}" for k, v in args.items())
    res = h.get("result_summary", "") or ""
    return f"{name}({arg_str}) -> {res}".strip()


def serialize_example(sample: dict, n_recent_actions: int = 3) -> str:
    """
    한 샘플을 인코더 입력 문자열로.
    설계 근거(실측):
      - 신호 대부분이 current_prompt에 있음 → 맨 앞에 배치(절대 안 잘리게).
      - 문맥은 소폭 기여 → 최근 action 몇 개 + workspace 핵심만 압축해서 뒤에.
      - full-dump는 오히려 성능 저하였음 → 최근 N개로 제한.
    """
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    hist = sample.get("history", []) or []
    actions = [h for h in hist if h.get("role") == "assistant_action"]

    parts = []
    # 1) 현재 발화 (최우선)
    parts.append("[PROMPT] " + (sample.get("current_prompt", "") or ""))

    # 2) 직전 사용자 발화 (있으면 맥락)
    users = [h for h in hist if h.get("role") == "user"]
    if users:
        parts.append("[LAST_USER] " + (users[-1].get("content", "") or ""))

    # 3) 최근 action 궤적 (오래된 것부터, 최근 N개만)
    if actions:
        recent = actions[-n_recent_actions:]
        traj = " | ".join(_fmt_action(a) for a in recent)
        parts.append("[RECENT] " + traj)

    # 4) workspace 상태 (압축)
    parts.append(
        "[STATE] "
        f"ci={ws.get('last_ci_status')} "
        f"dirty={ws.get('git_dirty')} "
        f"open={len(ws.get('open_files', []) or [])} "
        f"turn={meta.get('turn_index')} "
        f"tier={meta.get('user_tier')} "
        f"lang={meta.get('language_pref')}"
    )
    return "\n".join(parts)


def load_train(data_dir: str) -> tuple[list[str], list[str], list[int], list[str]]:
    """
    return: (ids, texts, labels(int), groups(session_id))
    train_labels.csv 를 id로 조인.
    """
    import csv
    samples = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    labels = {}
    with open(os.path.join(data_dir, "train_labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    ids, texts, ys, groups = [], [], [], []
    for s in samples:
        sid = s["id"]
        ids.append(sid)
        texts.append(serialize_example(s))
        ys.append(CLASS2ID[labels[sid]])
        groups.append(session_id(sid))
    return ids, texts, ys, groups


def load_test(data_dir: str) -> tuple[list[str], list[str]]:
    """제출용: (ids, texts). 라벨 없음."""
    samples = read_jsonl(os.path.join(data_dir, "test.jsonl"))
    ids = [s["id"] for s in samples]
    texts = [serialize_example(s) for s in samples]
    return ids, texts
