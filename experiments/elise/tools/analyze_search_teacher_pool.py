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
    out = {"macro_f1": macro_f1(y, pred), "accuracy": float((pred == y).mean())}
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


def blend(current, candidate, op, weight):
    if op == "prob":
        return (1 - weight) * current + weight * candidate
    if op == "logprob":
        return softmax((1 - weight) * logp(current) + weight * logp(candidate))
    raise ValueError(op)


def check_probs(name, arr, n):
    if arr.shape != (n, NUM_CLASSES):
        raise RuntimeError(f"{name}: bad shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name}: non-finite values")
    if arr.min() < -1e-6:
        raise RuntimeError(f"{name}: negative probabilities")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError(f"{name}: row sums are not probability-like")


def better(item, best):
    if best is None:
        return True
    score, count, weight, name, op = item
    b_score, b_count, b_weight, b_name, b_op = best
    tol = 1e-12
    if score > b_score + tol:
        return True
    if abs(score - b_score) <= tol:
        if count < b_count:
            return True
        if count == b_count:
            if weight < b_weight - tol:
                return True
            if abs(weight - b_weight) <= tol:
                if name < b_name:
                    return True
                if name == b_name and OPS.index(op) < OPS.index(b_op):
                    return True
    return False


def greedy_select(y, candidates, idx):
    names = sorted(candidates)
    best_start = None
    for name in names:
        score = macro_f1(y[idx], candidates[name][idx].argmax(axis=1))
        item = (score, 1, 0.0, name, "start")
        if better(item, best_start):
            best_start = item
    selected = [best_start[3]]
    current = candidates[selected[0]].copy()
    current_score = best_start[0]
    trajectory = [{"event": "start", "candidate": selected[0], "selection_macro_f1": float(current_score)}]
    remaining = [name for name in names if name not in selected]
    while remaining:
        best = None
        best_probs = None
        for name in remaining:
            for op in OPS:
                for weight in WEIGHTS:
                    probs = blend(current, candidates[name], op, weight)
                    score = macro_f1(y[idx], probs[idx].argmax(axis=1))
                    item = (score, len(selected) + 1, weight, name, op)
                    if better(item, best):
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
    return current, {"start": selected[0], "selected": selected, "inner_score": float(current_score), "trajectory": trajectory}


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
                local_best = (best, current)
                for value in [current, np.clip(current + step, -0.5, 0.5), np.clip(current - step, -0.5, 0.5)]:
                    trial = bias.copy()
                    trial[c] = value
                    trial -= trial.mean()
                    trial = np.clip(trial, -0.5, 0.5)
                    s = macro_f1(y[idx], (score[idx] + trial).argmax(axis=1))
                    if s > local_best[0] + 1e-12:
                        local_best = (s, value)
                if local_best[0] > best + 1e-12:
                    bias[c] = local_best[1]
                    bias -= bias.mean()
                    bias = np.clip(bias, -0.5, 0.5)
                    best = local_best[0]
                    improved = True
    return bias, float(best)


def run_nested(y, fold_ids, candidates):
    raw_oof = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    bias_oof = np.zeros_like(raw_oof)
    choices = {}
    for fold in sorted(np.unique(fold_ids)):
        select_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]
        probs, info = greedy_select(y, candidates, select_idx)
        raw_oof[test_idx] = probs[test_idx].astype(np.float32)
        bias, inner = fit_bias(y, probs, select_idx)
        bias_oof[test_idx] = softmax(logp(probs[test_idx]) + bias).astype(np.float32)
        choices[f"fold{int(fold)}"] = {
            **info,
            "bias_inner_score": inner,
            "outer_raw_score": macro_f1(y[test_idx], raw_oof[test_idx].argmax(axis=1)),
            "outer_bias_score": macro_f1(y[test_idx], bias_oof[test_idx].argmax(axis=1)),
            "bias": {CLASSES[i]: float(bias[i]) for i in range(NUM_CLASSES)},
        }
        print(
            f"fold {fold}: selected={info['selected']} "
            f"raw={choices[f'fold{int(fold)}']['outer_raw_score']:.6f} "
            f"bias={choices[f'fold{int(fold)}']['outer_bias_score']:.6f}",
            flush=True,
        )
    return raw_oof, bias_oof, choices


def paired(y, pred_a, pred_b, a_name, b_name):
    a_only = (pred_a == y) & (pred_b != y)
    b_only = (pred_a != y) & (pred_b == y)
    return {
        "a": a_name,
        "b": b_name,
        "disagreement_count": int((pred_a != pred_b).sum()),
        "disagreement_ratio": float((pred_a != pred_b).mean()),
        "a_only_correct": int(a_only.sum()),
        "b_only_correct": int(b_only.sum()),
        "net_b_gain": int(b_only.sum() - a_only.sum()),
        "oracle_either_accuracy": float(((pred_a == y) | (pred_b == y)).mean()),
    }


def per_class_report(y, ref_pred, pred):
    ref = per_class_f1(y, ref_pred)
    cur = per_class_f1(y, pred)
    return [
        {
            "class": CLASSES[i],
            "support": int((y == i).sum()),
            "ref_f1": float(ref[i, 2]),
            "candidate_f1": float(cur[i, 2]),
            "delta_vs_ref": float(cur[i, 2] - ref[i, 2]),
        }
        for i in range(NUM_CLASSES)
    ]


def load_candidates(paths, n):
    out = {}
    for name, path in paths.items():
        arr = np.load(path)
        check_probs(name, arr, n)
        out[name] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="search_teacher_pool")
    args = ap.parse_args()

    ids, _, y, _ = load_train("data/original_data")
    y = np.array(y, dtype=np.int64)
    train_ids = np.load("oof/train_ids.npy", allow_pickle=True)
    if list(train_ids) != list(ids):
        raise RuntimeError("train id order mismatch")
    fold_ids = np.load("oof/sgkf5_split_seed42_fold_ids.npy")
    if fold_ids.shape != (len(y),):
        raise RuntimeError("bad fold id shape")

    paths = {
        "baseline": "oof/fields_mmbert_ep3.npy",
        "flow42": "oof/fields_flow_ep3.npy",
        "fields_seed45": "oof/fields_mmbert_seed45_probs.npy",
        "fields_seed46": "oof/fields_mmbert_seed46_probs.npy",
        "fields_seed47": "oof/fields_mmbert_seed47_probs.npy",
        "flow45": "oof/fields_flow_seed45_ep3_probs.npy",
        "flow47": "oof/fields_flow_seed47_ep3_probs.npy",
    }
    candidates = load_candidates(paths, len(y))
    baseline_pred = candidates["baseline"].argmax(axis=1)
    old_teacher = np.load("oof/current_greedy_all_available_probs.npy")
    check_probs("current_greedy_all_available", old_teacher, len(y))
    old_teacher_pred = old_teacher.argmax(axis=1)

    singles = {name: evaluate(y, probs, baseline_pred) for name, probs in candidates.items()}
    fixed_probs, fixed_info = greedy_select(y, candidates, np.arange(len(y)))
    fixed_metrics = evaluate(y, fixed_probs, baseline_pred)
    raw_nested, bias_nested, nested_choices = run_nested(y, fold_ids, candidates)
    raw_metrics = evaluate(y, raw_nested, baseline_pred)
    bias_metrics = evaluate(y, bias_nested, baseline_pred)

    selection_counts = Counter()
    for choice in nested_choices.values():
        for name in choice["selected"]:
            selection_counts[name] += 1
    all_correct = np.column_stack([arr.argmax(axis=1) == y for arr in candidates.values()])
    summary = {
        "candidate_paths": paths,
        "singles": singles,
        "previous_fixed_teacher": evaluate(y, old_teacher, baseline_pred),
        "fixed_greedy": {"metrics": fixed_metrics, **fixed_info},
        "fixed_delta_vs_previous_teacher": fixed_metrics["macro_f1"] - macro_f1(y, old_teacher_pred),
        "nested_raw_metrics": raw_metrics,
        "nested_bias_metrics": bias_metrics,
        "nested_delta_raw_vs_previous_nested_bias": raw_metrics["macro_f1"] - 0.7816492429205367,
        "nested_delta_bias_vs_previous_nested_bias": bias_metrics["macro_f1"] - 0.7816492429205367,
        "nested_fold_choices": nested_choices,
        "nested_selection_counts": dict(selection_counts),
        "paired_previous_fixed_vs_new_fixed": paired(y, old_teacher_pred, fixed_probs.argmax(axis=1), "previous_fixed", "new_fixed"),
        "paired_baseline_vs_new_fixed": paired(y, baseline_pred, fixed_probs.argmax(axis=1), "baseline", "new_fixed"),
        "paired_previous_fixed_vs_nested_raw": paired(y, old_teacher_pred, raw_nested.argmax(axis=1), "previous_fixed", "nested_raw"),
        "oracle_any_candidate_accuracy": float(all_correct.any(axis=1).mean()),
        "per_class_fixed_vs_previous": per_class_report(y, old_teacher_pred, fixed_probs.argmax(axis=1)),
        "per_class_nested_raw_vs_previous": per_class_report(y, old_teacher_pred, raw_nested.argmax(axis=1)),
    }

    Path("results").mkdir(exist_ok=True)
    Path("oof").mkdir(exist_ok=True)
    json_path = f"results/{args.out_prefix}.json"
    md_path = f"results/{args.out_prefix}.md"
    fixed_path = f"oof/{args.out_prefix}_fixed_probs.npy"
    nested_path = f"oof/{args.out_prefix}_nested_probs.npy"
    nested_bias_path = f"oof/{args.out_prefix}_nested_bias_probs.npy"
    np.save(fixed_path, fixed_probs.astype(np.float32))
    np.save(nested_path, raw_nested.astype(np.float32))
    np.save(nested_bias_path, bias_nested.astype(np.float32))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["# Search Teacher Pool Summary\n\n"]
    lines.append("## Singles\n\n| candidate | macro_f1 | accuracy |\n|---|---:|---:|\n")
    for name, row in sorted(singles.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True):
        lines.append(f"| {name} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} |\n")
    lines.append("\n## Fixed Greedy\n\n")
    lines.append(f"- Macro-F1: {fixed_metrics['macro_f1']:.6f}\n")
    lines.append(f"- Accuracy: {fixed_metrics['accuracy']:.6f}\n")
    lines.append(f"- Delta vs previous fixed teacher: {summary['fixed_delta_vs_previous_teacher']:+.6f}\n")
    lines.append(f"- Selected: `{fixed_info['selected']}`\n")
    lines.append("\n## Fully Nested\n\n")
    lines.append(f"- Raw Macro-F1: {raw_metrics['macro_f1']:.6f}\n")
    lines.append(f"- Raw Accuracy: {raw_metrics['accuracy']:.6f}\n")
    lines.append(f"- Bias Macro-F1: {bias_metrics['macro_f1']:.6f}\n")
    lines.append(f"- Bias Accuracy: {bias_metrics['accuracy']:.6f}\n")
    lines.append(f"- Raw delta vs previous nested bias: {summary['nested_delta_raw_vs_previous_nested_bias']:+.6f}\n")
    lines.append(f"- Bias delta vs previous nested bias: {summary['nested_delta_bias_vs_previous_nested_bias']:+.6f}\n")
    lines.append(f"- Selection counts: `{dict(selection_counts)}`\n")
    lines.append("\n## Nested Fold Choices\n\n")
    for fold, choice in nested_choices.items():
        lines.append(f"- {fold}: selected={choice['selected']} raw={choice['outer_raw_score']:.6f} bias={choice['outer_bias_score']:.6f}\n")
    lines.append("\n## 4090 Promotion\n\n")
    lines.append("- `search4090_flow_last_action`: Fold0 ep3 delta +0.002682, status PROMOTE_FULL_CV.\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(json.dumps({
        "singles": singles,
        "fixed_greedy": summary["fixed_greedy"],
        "fixed_delta_vs_previous_teacher": summary["fixed_delta_vs_previous_teacher"],
        "nested_raw_metrics": raw_metrics,
        "nested_bias_metrics": bias_metrics,
        "nested_selection_counts": dict(selection_counts),
        "paths": {
            "json": json_path,
            "md": md_path,
            "fixed": fixed_path,
            "nested": nested_path,
            "nested_bias": nested_bias_path,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
