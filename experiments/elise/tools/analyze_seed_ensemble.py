import argparse
import glob
import json
import os
import re
from itertools import combinations

import numpy as np
from sklearn.metrics import f1_score

from src.data import NUM_CLASSES, load_train


def macro(y, pred):
    return f1_score(y, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_candidates(pattern):
    rows = []
    for prob_path in sorted(glob.glob(pattern)):
        if not prob_path.endswith("_probs.npy"):
            continue
        stem = prob_path[:-len("_probs.npy")]
        logits_path = stem + "_logits.npy"
        meta_path = stem + "_meta.json"
        if not os.path.exists(logits_path) or not os.path.exists(meta_path):
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        rows.append({
            "name": os.path.basename(stem),
            "probs_path": prob_path,
            "logits_path": logits_path,
            "meta_path": meta_path,
            "meta": meta,
        })
    return rows


def score_blend(y, names, arrays, mode):
    arr = np.mean([arrays[name] for name in names], axis=0)
    probs = softmax(arr) if mode == "logit" else arr
    return macro(y, probs.argmax(axis=1))


def greedy(y, candidates, arrays, mode):
    singles = [(score_blend(y, [c["name"]], arrays, mode), c["name"]) for c in candidates]
    singles.sort(reverse=True)
    selected = [singles[0][1]]
    current = arrays[selected[0]].copy()
    current_score = score_blend(y, selected, arrays, mode)
    trace = [{"selected": selected.copy(), "score": current_score, "event": "start"}]
    remaining = {c["name"] for c in candidates} - set(selected)
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    while remaining:
        best = None
        for name in sorted(remaining):
            for w in weights:
                blend = (1.0 - w) * current + w * arrays[name]
                probs = softmax(blend) if mode == "logit" else blend
                score = macro(y, probs.argmax(axis=1))
                item = (score, name, w, score - current_score)
                if best is None or item > best:
                    best = item
        score, name, weight, delta = best
        trace.append({"candidate": name, "weight": weight, "score": score, "delta": delta, "strong_accept": delta >= 0.001})
        if delta <= 0:
            break
        current = (1.0 - weight) * current + weight * arrays[name]
        current_score = score
        selected.append(name)
        remaining.remove(name)
    return {"mode": mode, "selected": selected, "score": current_score, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--pattern", default="oof/fields_mmbert_seed*_probs.npy")
    ap.add_argument("--out", default="results/seed_ensemble_summary.json")
    args = ap.parse_args()
    _, _, y, _ = load_train(args.data_dir)
    y = np.array(y, dtype=np.int64)
    candidates = load_candidates(args.pattern)
    if not candidates:
        raise RuntimeError(f"no candidates found for {args.pattern}")
    hashes = {(c["meta"].get("train_ids_sha256"), c["meta"].get("fold_assignment_sha256")) for c in candidates}
    if len(hashes) != 1:
        raise RuntimeError("candidate train id / fold assignment hash mismatch")
    prob_arrays, logit_arrays = {}, {}
    singles = []
    for c in candidates:
        probs = np.load(c["probs_path"])
        logits = np.load(c["logits_path"])
        if probs.shape != (len(y), NUM_CLASSES) or logits.shape != (len(y), NUM_CLASSES):
            raise RuntimeError(f"bad shape for {c['name']}")
        if not np.isfinite(probs).all() or not np.isfinite(logits).all():
            raise RuntimeError(f"non-finite values for {c['name']}")
        prob_arrays[c["name"]] = probs
        logit_arrays[c["name"]] = logits
        singles.append({"name": c["name"], "macro_f1": macro(y, probs.argmax(axis=1)), "meta": c["meta_path"]})
    pairwise = []
    for a, b in combinations([c["name"] for c in candidates], 2):
        pairwise.append({
            "names": [a, b],
            "prob_mean_macro_f1": score_blend(y, [a, b], prob_arrays, "prob"),
            "logit_mean_macro_f1": score_blend(y, [a, b], logit_arrays, "logit"),
        })
    all_names = [c["name"] for c in candidates]
    summary = {
        "candidates": singles,
        "pairwise": pairwise,
        "all_prob_mean_macro_f1": score_blend(y, all_names, prob_arrays, "prob"),
        "all_logit_mean_macro_f1": score_blend(y, all_names, logit_arrays, "logit"),
        "greedy_prob": greedy(y, candidates, prob_arrays, "prob"),
        "greedy_logit": greedy(y, candidates, logit_arrays, "logit"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
