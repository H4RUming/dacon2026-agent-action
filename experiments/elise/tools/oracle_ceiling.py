import argparse
import glob
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import NUM_CLASSES, load_train


def load_probs(paths: list[str]) -> list[np.ndarray]:
    probs = []
    for path in paths:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != NUM_CLASSES:
            raise ValueError(f"{path}: expected [N, {NUM_CLASSES}], got {arr.shape}")
        probs.append(arr)
    n = probs[0].shape[0]
    for path, arr in zip(paths, probs):
        if arr.shape[0] != n:
            raise ValueError(f"{path}: expected N={n}, got {arr.shape[0]}")
    return probs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/original_data")
    parser.add_argument("--probs", nargs="+", default=None)
    parser.add_argument("--glob", default="oof/*.npy")
    args = parser.parse_args()

    paths = args.probs or sorted(glob.glob(args.glob))
    if not paths:
        print(f"no probability files found: {args.glob}", file=sys.stderr)
        return 1

    _, _, y, _ = load_train(args.data_dir)
    y = np.array(y)
    probs = load_probs(paths)
    if probs[0].shape[0] != len(y):
        raise ValueError(f"prob N={probs[0].shape[0]} does not match labels N={len(y)}")

    preds = [p.argmax(1) for p in probs]
    correct = [(pred == y) for pred in preds]

    print("files:")
    for i, path in enumerate(paths):
        score = f1_score(y, preds[i], labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
        acc = correct[i].mean()
        print(f"  m{i + 1}: {path} macro_f1={score:.6f} acc={acc:.6f}")

    agree_all = np.ones(len(y), dtype=bool)
    for pred in preds[1:]:
        agree_all &= pred == preds[0]
    disagree = ~agree_all

    print(f"all_agree_ratio {agree_all.mean():.6f}")
    print(f"disagree_ratio {disagree.mean():.6f}")

    if disagree.any():
        for i, ok in enumerate(correct):
            print(f"m{i + 1}_acc_on_disagree {ok[disagree].mean():.6f}")
    else:
        print("acc_on_disagree skipped: no disagreements")

    oracle_hit = np.zeros(len(y), dtype=bool)
    for ok in correct:
        oracle_hit |= ok
    print(f"top1_oracle_acc {oracle_hit.mean():.6f}")

    for i, ok in enumerate(correct):
        others_correct = np.zeros(len(y), dtype=bool)
        for j, other_ok in enumerate(correct):
            if i != j:
                others_correct |= other_ok
        unique_correct = ok & (~others_correct)
        print(f"m{i + 1}_unique_correct {unique_correct.mean():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
