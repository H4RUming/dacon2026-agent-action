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


def serialize_example(sample: dict, n_recent_actions: int = 999) -> str:
    """문맥 전체 버전: history 전량 + 모든 args + result. (max_len으로 자연 truncate)"""
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    hist = sample.get("history", []) or []

    parts = []
    parts.append("[PROMPT] " + (sample.get("current_prompt", "") or ""))
    parts.append(
        "[STATE] "
        f"ci={ws.get('last_ci_status')} dirty={ws.get('git_dirty')} "
        f"open={len(ws.get('open_files', []) or [])} turn={meta.get('turn_index')} "
        f"tier={meta.get('user_tier')} lang={meta.get('language_pref')} "
        f"loc={ws.get('loc')} budget={meta.get('budget_tokens_remaining')}"
    )
    # 열린 파일 목록도 (경로가 read/edit 신호일 수 있음)
    of = ws.get("open_files", []) or []
    if of:
        parts.append("[OPEN] " + ", ".join(of))
    # history 전체를 시간순으로
    parts.append("[HISTORY]")
    for h in hist:
        if h.get("role") == "user":
            parts.append("U: " + (h.get("content", "") or ""))
        else:
            args = h.get("args", {}) or {}
            arg_str = " ".join(f"{k}={v}" for k, v in args.items())
            parts.append(f"A: {h.get('name','')}({arg_str}) -> {h.get('result_summary','') or ''}")
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
