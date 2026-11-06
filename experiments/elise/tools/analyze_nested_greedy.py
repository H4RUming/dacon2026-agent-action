import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import CLASSES, NUM_CLASSES, load_train


EPS = 1e-12
WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
OPS = ["prob", "logprob"]


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logp(p):
    return np.log(np.clip(p, EPS, 1.0))


def per_class_f1(y, pred):
    rows = []
    for c in range(NUM_CLASSES):
        tp = int(((y == c) & (pred == c)).sum())
        fp = int(((y != c) & (pred == c)).sum())
        fn = int(((y == c) & (pred != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append((p, r, f1, int((y == c).sum())))
    return np.array(rows, dtype=float)


def macro_f1(y, pred):
    return float(per_class_f1(y, pred)[:, 2].mean())


def evaluate(y, probs, baseline_pred=None):
    pred = probs.argmax(axis=1)
    out = {
        "macro_f1": macro_f1(y, pred),
        "accuracy": float((pred == y).mean()),
    }
    if baseline_pred is not None:
        out.update({
            "delta_macro_vs_baseline": out["macro_f1"] - macro_f1(y, baseline_pred),
            "delta_acc_vs_baseline": out["accuracy"] - float((baseline_pred == y).mean()),
            "changed_vs_baseline": int((pred != baseline_pred).sum()),
            "baseline_wrong_to_correct": int(((baseline_pred != y) & (pred == y)).sum()),
            "baseline_correct_to_wrong": int(((baseline_pred == y) & (pred != y)).sum()),
        })
        out["net_correct_gain_vs_baseline"] = out["baseline_wrong_to_correct"] - out["baseline_correct_to_wrong"]
    return out


def blend_probs(current, candidate, op, weight):
    if op == "prob":
        return (1.0 - weight) * current + weight * candidate
    if op == "logprob":
        return softmax((1.0 - weight) * logp(current) + weight * logp(candidate))
    raise ValueError(op)


def better_score(candidate, best):
    if best is None:
        return True
    score, model_count, weight, name, op = candidate
    b_score, b_model_count, b_weight, b_name, b_op = best
    tol = 1e-12
    if score > b_score + tol:
        return True
    if abs(score - b_score) <= tol:
        if model_count < b_model_count:
            return True
        if model_count == b_model_count:
            if weight < b_weight - tol:
                return True
            if abs(weight - b_weight) <= tol:
                if name < b_name:
                    return True
                if name == b_name and OPS.index(op) < OPS.index(b_op):
                    return True
    return False


def greedy_select(y, candidates, selection_idx):
    names = sorted(candidates)
    best_start = None
    for name in names:
        score = macro_f1(y[selection_idx], candidates[name][selection_idx].argmax(axis=1))
        item = (score, 1, 0.0, name, "start")
        if better_score(item, best_start):
            best_start = item
    current_name = best_start[3]
    selected = [current_name]
    current = candidates[current_name].copy()
    current_score = best_start[0]
    trajectory = [{
        "event": "start",
        "candidate": current_name,
        "selection_macro_f1": float(current_score),
    }]
    remaining = [name for name in names if name not in selected]
    while remaining:
        best = None
        best_probs = None
        for name in remaining:
            for op in OPS:
                for weight in WEIGHTS:
                    probs = blend_probs(current, candidates[name], op, weight)
                    score = macro_f1(y[selection_idx], probs[selection_idx].argmax(axis=1))
                    item = (score, len(selected) + 1, weight, name, op)
                    if better_score(item, best):
                        best = item
                        best_probs = probs
        score, _, weight, name, op = best
        delta = score - current_score
        trajectory.append({
            "event": "try_add",
            "candidate": name,
            "operation": op,
            "weight": float(weight),
            "selection_macro_f1": float(score),
            "delta": float(delta),
        })
        if delta <= 0:
            break
        selected.append(name)
        remaining = [n for n in remaining if n != name]
        current = best_probs
        current_score = score
    return current, {
        "start": selected[0],
        "selected": selected,
        "inner_score": float(current_score),
        "trajectory": trajectory,
    }


def fit_bias(y, probs, idx):
    score = logp(probs)
    bias = np.zeros(NUM_CLASSES, dtype=np.float64)
    best = macro_f1(y[idx], (score[idx] + bias).argmax(axis=1))
    for step in [0.10, 0.05, 0.02, 0.01]:
        improved = True
        while improved:
            improved = False
            for c in range(NUM_CLASSES):
                current = bias[c]
                candidates = [current, np.clip(current + step, -0.5, 0.5), np.clip(current - step, -0.5, 0.5)]
                local_best_score = best
                local_best_value = current
                for value in candidates:
                    trial = bias.copy()
                    trial[c] = value
                    trial -= trial.mean()
                    trial = np.clip(trial, -0.5, 0.5)
                    s = macro_f1(y[idx], (score[idx] + trial).argmax(axis=1))
                    if s > local_best_score + 1e-12:
                        local_best_score = s
                        local_best_value = value
                if local_best_score > best + 1e-12:
                    bias[c] = local_best_value
                    bias -= bias.mean()
                    bias = np.clip(bias, -0.5, 0.5)
                    best = local_best_score
                    improved = True
    return bias, float(best)


def check_probs(name, arr, n):
    if arr.shape != (n, NUM_CLASSES):
        raise RuntimeError(f"{name}: bad shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name}: non-finite values")
    if arr.min() < -1e-6:
        raise RuntimeError(f"{name}: negative probabilities")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError(f"{name}: row sums are not probabilities")


def load_candidates(include_flow45=False):
    paths = {
        "baseline": "oof/fields_mmbert_ep3.npy",
        "flow42": "oof/fields_flow_ep3.npy",
        "seed45": "oof/fields_mmbert_seed45_probs.npy",
        "seed46": "oof/fields_mmbert_seed46_probs.npy",
        "seed47": "oof/fields_mmbert_seed47_probs.npy",
    }
    if include_flow45:
        paths["flow45"] = "oof/fields_flow_seed45_ep3_probs.npy"
    out = {}
    for name, path in paths.items():
        if include_flow45 or name != "flow45":
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            out[name] = np.load(path)
    return out, paths


def paired(y, pred_a, pred_b, a_name, b_name):
    both = (pred_a == y) & (pred_b == y)
    a_only = (pred_a == y) & (pred_b != y)
    b_only = (pred_a != y) & (pred_b == y)
    wrong = (pred_a != y) & (pred_b != y)
    return {
        "a": a_name,
        "b": b_name,
        "disagreement_count": int((pred_a != pred_b).sum()),
        "disagreement_ratio": float((pred_a != pred_b).mean()),
        "both_correct": int(both.sum()),
        "a_only_correct": int(a_only.sum()),
        "b_only_correct": int(b_only.sum()),
        "net_b_gain": int(b_only.sum() - a_only.sum()),
        "both_wrong": int(wrong.sum()),
        "oracle_either_accuracy": float(((pred_a == y) | (pred_b == y)).mean()),
    }


def per_class_report(y, baseline_pred, candidate_pred):
    b = per_class_f1(y, baseline_pred)
    c = per_class_f1(y, candidate_pred)
    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append({
            "class": cls,
            "support": int((y == i).sum()),
            "baseline_f1": float(b[i, 2]),
            "candidate_f1": float(c[i, 2]),
            "delta_vs_baseline": float(c[i, 2] - b[i, 2]),
        })
    return rows


def run_nested(y, fold_ids, candidates):
    n = len(y)
    raw_oof = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    bias_oof = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    fold_choices = {}
    for fold in sorted(np.unique(fold_ids)):
        selection_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]
        probs, info = greedy_select(y, candidates, selection_idx)
        raw_oof[test_idx] = probs[test_idx].astype(np.float32)
        bias, bias_inner = fit_bias(y, probs, selection_idx)
        adjusted = softmax(logp(probs[test_idx]) + bias)
        bias_oof[test_idx] = adjusted.astype(np.float32)
        fold_choices[f"fold{int(fold)}"] = {
            **info,
            "bias_inner_score": bias_inner,
            "outer_raw_score": macro_f1(y[test_idx], raw_oof[test_idx].argmax(axis=1)),
            "outer_bias_score": macro_f1(y[test_idx], bias_oof[test_idx].argmax(axis=1)),
            "bias": {CLASSES[i]: float(bias[i]) for i in range(NUM_CLASSES)},
        }
        print(f"fold {fold}: selected={info['selected']} raw={fold_choices[f'fold{int(fold)}']['outer_raw_score']:.6f} bias={fold_choices[f'fold{int(fold)}']['outer_bias_score']:.6f}", flush=True)
    return raw_oof, bias_oof, fold_choices


def classify(score):
    if score >= 0.7800:
        return "STRONG"
    if score >= 0.7790:
        return "USEFUL"
    if score >= 0.7775:
        return "CAUTION"
    return "OVERFIT_RISK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include_flow45", action="store_true")
    ap.add_argument("--out_prefix", default="current")
    args = ap.parse_args()
    ids, _, y, _ = load_train("data/original_data")
    y = np.array(y, dtype=np.int64)
    n = len(y)
    fold_ids = np.load("oof/sgkf5_split_seed42_fold_ids.npy")
    train_ids = np.load("oof/train_ids.npy", allow_pickle=True)
    if list(train_ids) != list(ids):
        raise RuntimeError("train id order mismatch")
    candidates, paths = load_candidates(args.include_flow45)
    for name, arr in candidates.items():
        check_probs(name, arr, n)
    baseline_pred = candidates["baseline"].argmax(axis=1)
    singles = {name: evaluate(y, arr, baseline_pred) for name, arr in candidates.items()}
    raw, bias, choices = run_nested(y, fold_ids, candidates)
    raw_metrics = evaluate(y, raw, baseline_pred)
    bias_metrics = evaluate(y, bias, baseline_pred)
    selection_counts = Counter()
    for c in choices.values():
        for name in c["selected"]:
            selection_counts[name] += 1
    summary = {
        "candidate_paths": paths,
        "singles": singles,
        "raw_metrics": raw_metrics,
        "bias_metrics": bias_metrics,
        "classification": classify(raw_metrics["macro_f1"]),
        "fold_choices": choices,
        "selection_counts": dict(selection_counts),
        "paired_baseline_raw": paired(y, baseline_pred, raw.argmax(axis=1), "baseline", "nested_raw"),
        "paired_baseline_bias": paired(y, baseline_pred, bias.argmax(axis=1), "baseline", "nested_bias"),
        "per_class_raw": per_class_report(y, baseline_pred, raw.argmax(axis=1)),
        "per_class_bias": per_class_report(y, baseline_pred, bias.argmax(axis=1)),
        "oracle_any_accuracy": float(np.column_stack([(arr.argmax(axis=1) == y) for arr in candidates.values()]).any(axis=1).mean()),
    }
    Path("results").mkdir(exist_ok=True)
    Path("oof").mkdir(exist_ok=True)
    if args.include_flow45:
        json_path = "results/fully_nested_greedy_with_flow45.json"
        md_path = "results/fully_nested_greedy_with_flow45.md"
        raw_path = "oof/nested_greedy_with_flow45_probs.npy"
        bias_path = "oof/nested_greedy_with_flow45_bias_probs.npy"
    else:
        json_path = "results/fully_nested_greedy_current.json"
        md_path = "results/fully_nested_greedy_current.md"
        raw_path = "oof/nested_greedy_all_available_probs.npy"
        bias_path = "oof/nested_greedy_all_available_bias_probs.npy"
    np.save(raw_path, raw.astype(np.float32))
    np.save(bias_path, bias.astype(np.float32))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = []
    md.append("# Fully Nested Greedy\n\n")
    md.append(f"- Raw Macro-F1: {raw_metrics['macro_f1']:.6f}\n")
    md.append(f"- Raw Accuracy: {raw_metrics['accuracy']:.6f}\n")
    md.append(f"- Bias Macro-F1: {bias_metrics['macro_f1']:.6f}\n")
    md.append(f"- Bias Accuracy: {bias_metrics['accuracy']:.6f}\n")
    md.append(f"- Classification: {summary['classification']}\n")
    md.append(f"- Selection counts: `{summary['selection_counts']}`\n\n")
    md.append("## Fold Choices\n")
    for fold, choice in choices.items():
        md.append(f"- {fold}: start={choice['start']} selected={choice['selected']} raw={choice['outer_raw_score']:.6f} bias={choice['outer_bias_score']:.6f}\n")
    md.append("\n## Per-Class Raw Delta\n")
    md.append("| class | support | delta |\n|---|---:|---:|\n")
    for row in sorted(summary["per_class_raw"], key=lambda r: r["delta_vs_baseline"]):
        md.append(f"| {row['class']} | {row['support']} | {row['delta_vs_baseline']:+.6f} |\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"saved {json_path}")
    print(f"saved {md_path}")
    print(f"saved {raw_path}")
    print(f"saved {bias_path}")
    print(json.dumps({
        "raw_metrics": raw_metrics,
        "bias_metrics": bias_metrics,
        "classification": summary["classification"],
        "selection_counts": summary["selection_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
