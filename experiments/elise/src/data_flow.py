import csv
import os
from collections import Counter

from src.data import CLASS2ID, CLASSES, ID2CLASS, NUM_CLASSES, read_jsonl, serialize_example, session_id


LASTRES_RULES = {
    "fail": ["failed", "failure", "error", "exception", "traceback", "non-zero", "nonzero", "exit code 1", "exit status 1", "실패", "오류"],
    "notfound": ["not found", "no match", "no matches", "0 matches", "missing"],
    "pass": ["passed", "success", "successful", "ok", "exit code 0", "exit status 0"],
    "changed": ["updated", "modified", "changed", "created", "written", "wrote", "applied", "patched", "deleted", "removed"],
    "found": ["found", "match", "matches", "located"],
}


def turn_bucket_flow(turn) -> str:
    try:
        t = int(turn)
    except (TypeError, ValueError):
        return "unknown"
    if t <= 0:
        return "0"
    if t == 1:
        return "1"
    if t == 2:
        return "2"
    if t == 3:
        return "3"
    if t <= 5:
        return "4-5"
    if t <= 7:
        return "6-7"
    if t <= 11:
        return "8-11"
    return "12+"


def assistant_actions(sample: dict) -> list[dict]:
    return [
        h for h in (sample.get("history", []) or [])
        if h.get("role") == "assistant_action"
    ]


def prev3_flow(sample: dict) -> str:
    names = [(h.get("name") or "__NONE__") for h in assistant_actions(sample)]
    last3 = names[-3:]
    padded = ["__NONE__"] * (3 - len(last3)) + last3
    return ">".join(padded)


def lastres_category(sample: dict) -> str:
    actions = assistant_actions(sample)
    if not actions:
        return "none"
    raw = actions[-1].get("result_summary")
    if raw is None:
        return "none"
    text = str(raw).lower().strip()
    if not text:
        return "none"
    for category in ["fail", "notfound", "pass", "changed", "found"]:
        if any(token in text for token in LASTRES_RULES[category]):
            return category
    return "other"


def flow_line(sample: dict) -> str:
    meta = sample.get("session_meta", {}) or {}
    return (
        f"[FLOW] prev3={prev3_flow(sample)} "
        f"turn={turn_bucket_flow(meta.get('turn_index'))} "
        f"lastres={lastres_category(sample)}"
    )


def serialize_example_flow(sample: dict) -> str:
    base = serialize_example(sample)
    first, sep, rest = base.partition("\n")
    if not sep:
        return first + "\n" + flow_line(sample)
    return first + "\n" + flow_line(sample) + "\n" + rest


def remove_flow_line(text: str) -> str:
    lines = text.split("\n")
    kept = [line for line in lines if not line.startswith("[FLOW] ")]
    return "\n".join(kept)


def flow_stats(samples: list[dict]) -> dict:
    prev_missing = Counter()
    turns = Counter()
    lastres = Counter()
    for sample in samples:
        parts = prev3_flow(sample).split(">")
        for i, part in enumerate(parts):
            if part == "__NONE__":
                prev_missing[f"pos{i + 1}"] += 1
        meta = sample.get("session_meta", {}) or {}
        turns[turn_bucket_flow(meta.get("turn_index"))] += 1
        lastres[lastres_category(sample)] += 1
    return {
        "prev3_missing_position_counts": dict(prev_missing),
        "turn_bucket_counts": dict(turns),
        "lastres_category_counts": dict(lastres),
    }


def load_train_flow(data_dir: str) -> tuple[list[str], list[str], list[int], list[str], list[dict]]:
    samples = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    labels = {}
    with open(os.path.join(data_dir, "train_labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    ids, texts, ys, groups = [], [], [], []
    for sample in samples:
        sid = sample["id"]
        ids.append(sid)
        texts.append(serialize_example_flow(sample))
        ys.append(CLASS2ID[labels[sid]])
        groups.append(session_id(sid))
    return ids, texts, ys, groups, samples


def load_test_flow(data_dir: str) -> tuple[list[str], list[str]]:
    samples = read_jsonl(os.path.join(data_dir, "test.jsonl"))
    ids = [sample["id"] for sample in samples]
    texts = [serialize_example_flow(sample) for sample in samples]
    return ids, texts
