import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import read_jsonl, serialize_example
from src.data_flow import flow_line, flow_stats, remove_flow_line, serialize_example_flow


def deterministic_train_subset(samples, n=100):
    if len(samples) <= n:
        return samples
    step = (len(samples) - 1) / float(n - 1)
    return [samples[round(i * step)] for i in range(n)]


def check_samples(name, samples):
    mismatches = 0
    for sample in samples:
        base = serialize_example(sample)
        flow = serialize_example_flow(sample)
        restored = remove_flow_line(flow)
        if restored != base:
            mismatches += 1
    print(f"{name}_checked: {len(samples)}", flush=True)
    return mismatches


def main() -> int:
    data_dir = "data/original_data"
    train = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    test_path = os.path.join(data_dir, "test.jsonl")
    test = read_jsonl(test_path)[:5] if os.path.exists(test_path) else []
    checked = deterministic_train_subset(train, 100) + test

    mismatch = 0
    mismatch += check_samples("train", deterministic_train_subset(train, 100))
    if test:
        mismatch += check_samples("test", test)
    print(f"flow_base_mismatch: {mismatch}", flush=True)

    print("sample FLOW lines:", flush=True)
    for sample in checked[:5]:
        print("  " + flow_line(sample), flush=True)
    stats = flow_stats(checked)
    print("prev3_missing_position_counts:", stats["prev3_missing_position_counts"], flush=True)
    print("lastres_category_counts:", stats["lastres_category_counts"], flush=True)
    print("turn_bucket_counts:", stats["turn_bucket_counts"], flush=True)
    print("lastres_rules_order: none, fail, notfound, pass, changed, found, other", flush=True)
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
