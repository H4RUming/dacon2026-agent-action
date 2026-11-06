import csv
import os
from collections import Counter

from src.data import CLASS2ID, read_jsonl, serialize_example, session_id
from src.data_flow import lastres_category, prev3_flow, turn_bucket_flow


def assistant_actions(sample: dict) -> list[dict]:
    return [
        h for h in (sample.get("history", []) or [])
        if h.get("role") == "assistant_action"
    ]


def prevn_flow(sample: dict, n: int) -> str:
    names = [(h.get("name") or "__NONE__") for h in assistant_actions(sample)]
    last = names[-n:]
    padded = ["__NONE__"] * (n - len(last)) + last
    return ">".join(padded)


def last_action(sample: dict) -> str:
    actions = assistant_actions(sample)
    if not actions:
        return "__NONE__"
    return actions[-1].get("name") or "__NONE__"


def ci_status(sample: dict) -> str:
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    raw = ws.get("last_ci_status")
    if raw is None or str(raw).strip() == "":
        return "unknown"
    return str(raw).strip().replace(" ", "_")


def flow_line_variant(sample: dict, variant: str) -> str:
    meta = sample.get("session_meta", {}) or {}
    if variant == "flow_exact":
        return (
            f"[FLOW] prev3={prev3_flow(sample)} "
            f"turn={turn_bucket_flow(meta.get('turn_index'))} "
            f"lastres={lastres_category(sample)}"
        )
    if variant == "flow_last_action":
        return (
            f"[FLOW] prev3={prev3_flow(sample)} "
            f"last_action={last_action(sample)} "
            f"turn={turn_bucket_flow(meta.get('turn_index'))} "
            f"lastres={lastres_category(sample)}"
        )
    if variant == "flow_prev4":
        return (
            f"[FLOW] prev4={prevn_flow(sample, 4)} "
            f"turn={turn_bucket_flow(meta.get('turn_index'))} "
            f"lastres={lastres_category(sample)}"
        )
    if variant == "flow_last_ci_status":
        return (
            f"[FLOW] prev3={prev3_flow(sample)} "
            f"turn={turn_bucket_flow(meta.get('turn_index'))} "
            f"lastres={lastres_category(sample)} "
            f"last_ci_status={ci_status(sample)}"
        )
    raise ValueError(f"unknown FLOW variant: {variant}")


def serialize_example_flow_variant(sample: dict, variant: str) -> str:
    base = serialize_example(sample)
    first, sep, rest = base.partition("\n")
    line = flow_line_variant(sample, variant)
    if not sep:
        return first + "\n" + line
    return first + "\n" + line + "\n" + rest


def variant_stats(samples: list[dict]) -> dict:
    last_action_counts = Counter(last_action(sample) for sample in samples)
    ci_counts = Counter(ci_status(sample) for sample in samples)
    prev4_missing = Counter()
    for sample in samples:
        for i, part in enumerate(prevn_flow(sample, 4).split(">")):
            if part == "__NONE__":
                prev4_missing[f"pos{i + 1}"] += 1
    return {
        "last_action_counts": dict(last_action_counts),
        "ci_status_counts": dict(ci_counts),
        "prev4_missing_position_counts": dict(prev4_missing),
    }


def load_train_flow_variant(data_dir: str, variant: str) -> tuple[list[str], list[str], list[int], list[str], list[dict]]:
    samples = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    labels = {}
    with open(os.path.join(data_dir, "train_labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    ids, texts, ys, groups = [], [], [], []
    for sample in samples:
        sid = sample["id"]
        ids.append(sid)
        texts.append(serialize_example_flow_variant(sample, variant))
        ys.append(CLASS2ID[labels[sid]])
        groups.append(session_id(sid))
    return ids, texts, ys, groups, samples
