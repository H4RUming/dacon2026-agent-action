import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import CLASSES, CLASS2ID, NUM_CLASSES, load_train


BASELINE_PATH = "oof/fields_mmbert_ep3.npy"
FLOW_PATH = "oof/fields_flow_ep3.npy"
SEED45_PROBS_PATH = "oof/fields_mmbert_seed45_probs.npy"
SEED45_LOGITS_PATH = "oof/fields_mmbert_seed45_logits.npy"
FOLD_PATH = "oof/sgkf5_split_seed42_fold_ids.npy"
TRAIN_IDS_PATH = "oof/train_ids.npy"


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logp(p):
    return np.log(np.clip(p, 1e-12, 1.0))


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


def evaluate(name, y, probs, baseline_pred=None, seed45_pred=None):
    pred = probs.argmax(axis=1)
    row = {
        "name": name,
        "macro_f1": macro_f1(y, pred),
        "accuracy": float((pred == y).mean()),
    }
    if baseline_pred is not None:
        row.update({
            "delta_macro_vs_baseline": row["macro_f1"] - macro_f1(y, baseline_pred),
            "delta_acc_vs_baseline": row["accuracy"] - float((baseline_pred == y).mean()),
            "changed_vs_baseline": int((pred != baseline_pred).sum()),
            "baseline_wrong_to_correct": int(((baseline_pred != y) & (pred == y)).sum()),
            "baseline_correct_to_wrong": int(((baseline_pred == y) & (pred != y)).sum()),
        })
        row["net_correct_gain_vs_baseline"] = row["baseline_wrong_to_correct"] - row["baseline_correct_to_wrong"]
    if seed45_pred is not None:
        row["changed_vs_seed45"] = int((pred != seed45_pred).sum())
    return row


def check_probs(name, arr, n):
    if arr.shape != (n, NUM_CLASSES):
        raise RuntimeError(f"{name}: bad shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name}: non-finite values")
    if arr.min() < -1e-6:
        raise RuntimeError(f"{name}: negative probabilities")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError(f"{name}: row sums not probability-like")


def load_labels_by_id(data_dir):
    labels = {}
    with open(os.path.join(data_dir, "train_labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    return labels


def scan_probability_blend(y, a, b, prefix, baseline_pred):
    rows = []
    for w in np.linspace(0, 1, 101):
        probs = (1.0 - w) * a + w * b
        rows.append({"weight_b": float(w), **evaluate(f"{prefix}_w{w:.2f}", y, probs, baseline_pred)})
    return rows


def scan_logprob_blend(y, a_probs, b_probs, prefix, baseline_pred):
    la, lb = logp(a_probs), logp(b_probs)
    rows = []
    for w in np.linspace(0, 1, 101):
        probs = softmax((1.0 - w) * la + w * lb)
        rows.append({"weight_b": float(w), **evaluate(f"{prefix}_w{w:.2f}", y, probs, baseline_pred)})
    return rows


def tune_bias_grid(y, score, idx, step=0.05, rounds=2, clip=0.5):
    bias = np.zeros(NUM_CLASSES, dtype=np.float64)
    best = macro_f1(y[idx], (score[idx] + bias).argmax(axis=1))
    grid = np.arange(-clip, clip + 1e-9, step)
    for _ in range(rounds):
        improved = True
        while improved:
            improved = False
            for c in range(NUM_CLASSES):
                old = bias[c]
                local_best = (best, old)
                for v in grid:
                    bias[c] = v
                    s = macro_f1(y[idx], (score[idx] + bias).argmax(axis=1))
                    if s > local_best[0] + 1e-12:
                        local_best = (s, v)
                bias[c] = local_best[1]
                if local_best[0] > best + 1e-12:
                    best = local_best[0]
                    improved = True
    return bias, best


def apply_strict_bias(y, score, fold_ids):
    out = np.zeros_like(score)
    fold_rows = []
    biases = {}
    for fold in sorted(np.unique(fold_ids)):
        val = np.where(fold_ids == fold)[0]
        tr = np.where(fold_ids != fold)[0]
        bias, train_score = tune_bias_grid(y, score, tr)
        out[val] = score[val] + bias
        fold_rows.append({
            "fold": int(fold),
            "train_macro_f1_after_bias": float(train_score),
            "val_macro_f1_after_bias": macro_f1(y[val], out[val].argmax(axis=1)),
            "bias": {CLASSES[i]: float(bias[i]) for i in range(NUM_CLASSES)},
        })
        biases[int(fold)] = bias
    return softmax(out), fold_rows


def full_bias(y, score):
    bias, train_score = tune_bias_grid(y, score, np.arange(len(y)))
    return softmax(score + bias), {
        "train_macro_f1_after_bias": float(train_score),
        "bias": {CLASSES[i]: float(bias[i]) for i in range(NUM_CLASSES)},
    }


def paired(y, pred_a, pred_b, a_name, b_name):
    both = (pred_a == y) & (pred_b == y)
    a_only = (pred_a == y) & (pred_b != y)
    b_only = (pred_a != y) & (pred_b == y)
    wrong = (pred_a != y) & (pred_b != y)
    return {
        "a": a_name,
        "b": b_name,
        "disagreement_ratio": float((pred_a != pred_b).mean()),
        "both_correct": int(both.sum()),
        "a_only_correct": int(a_only.sum()),
        "b_only_correct": int(b_only.sum()),
        "net_b_gain": int(b_only.sum() - a_only.sum()),
        "both_wrong": int(wrong.sum()),
        "oracle_either_accuracy": float(((pred_a == y) | (pred_b == y)).mean()),
    }


def greedy_forward(y, candidates, baseline_pred):
    # Start at best single model, then add candidates by probability/logprob grids.
    singles = [(evaluate(name, y, probs, baseline_pred)["macro_f1"], name) for name, probs in candidates.items()]
    singles.sort(reverse=True)
    selected = [singles[0][1]]
    current_prob = candidates[selected[0]].copy()
    current_log = logp(candidates[selected[0]])
    current_score = macro_f1(y, current_prob.argmax(axis=1))
    trace = [{"event": "start", "selected": selected.copy(), "macro_f1": current_score}]
    remaining = [name for _, name in singles if name not in selected]
    while remaining:
        best = None
        for name in remaining:
            for mode in ["prob", "logprob"]:
                for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    if mode == "prob":
                        probs = (1.0 - w) * current_prob + w * candidates[name]
                    else:
                        probs = softmax((1.0 - w) * current_log + w * logp(candidates[name]))
                    score = macro_f1(y, probs.argmax(axis=1))
                    item = (score, name, mode, w, probs)
                    if best is None or item[:4] > best[:4]:
                        best = item
        score, name, mode, w, probs = best
        delta = score - current_score
        trace.append({"candidate": name, "mode": mode, "weight": w, "macro_f1": score, "delta": delta})
        if delta <= 0:
            break
        selected.append(name)
        remaining.remove(name)
        current_prob = probs
        current_log = logp(probs)
        current_score = score
    return current_prob, {"selected": selected, "macro_f1": current_score, "trace": trace}


def per_class_report(y, baseline_pred, pred):
    b = per_class_f1(y, baseline_pred)
    p = per_class_f1(y, pred)
    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append({
            "class": cls,
            "support": int((y == i).sum()),
            "baseline_f1": float(b[i, 2]),
            "candidate_f1": float(p[i, 2]),
            "delta_f1": float(p[i, 2] - b[i, 2]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--out_json", default="results/current_oof_ensemble_analysis.json")
    ap.add_argument("--out_md", default="results/current_oof_ensemble_report.md")
    args = ap.parse_args()

    ids, _, y, groups = load_train(args.data_dir)
    y = np.array(y, dtype=np.int64)
    n = len(y)
    artifacts = {
        "baseline": np.load(BASELINE_PATH),
        "flow": np.load(FLOW_PATH),
        "seed45": np.load(SEED45_PROBS_PATH),
    }
    seed45_logits = np.load(SEED45_LOGITS_PATH)
    fold_ids = np.load(FOLD_PATH)
    train_ids = np.load(TRAIN_IDS_PATH, allow_pickle=True)

    if list(train_ids) != list(ids):
        raise RuntimeError("train_ids.npy row order does not match src.data.load_train")
    if fold_ids.shape != (n,) or sorted(set(fold_ids.tolist())) != [0, 1, 2, 3, 4]:
        raise RuntimeError("bad fold assignment")
    for name, arr in artifacts.items():
        check_probs(name, arr, n)
    if seed45_logits.shape != (n, NUM_CLASSES) or not np.isfinite(seed45_logits).all():
        raise RuntimeError("bad seed45 logits")

    preds = {name: probs.argmax(axis=1) for name, probs in artifacts.items()}
    baseline_pred = preds["baseline"]
    seed45_pred = preds["seed45"]
    results = {
        "inventory": {
            "baseline": BASELINE_PATH,
            "flow": FLOW_PATH,
            "seed45_probs": SEED45_PROBS_PATH,
            "seed45_logits": SEED45_LOGITS_PATH,
            "fold_ids": FOLD_PATH,
            "train_ids": TRAIN_IDS_PATH,
            "class_order": CLASSES,
            "n": n,
        },
        "validation": {
            "row_order_matches_load_train": True,
            "fold_counts": {str(i): int((fold_ids == i).sum()) for i in range(5)},
        },
        "singles": [evaluate(name, y, probs, baseline_pred, seed45_pred) for name, probs in artifacts.items()],
    }

    # Required blend families.
    flow_seed_prob = scan_probability_blend(y, artifacts["flow"], artifacts["seed45"], "flow_seed45_prob", baseline_pred)
    flow_seed_log = scan_logprob_blend(y, artifacts["flow"], artifacts["seed45"], "flow_seed45_logprob", baseline_pred)
    flow_base_prob = scan_probability_blend(y, artifacts["flow"], artifacts["baseline"], "flow_baseline_prob", baseline_pred)
    flow_base_log = scan_logprob_blend(y, artifacts["flow"], artifacts["baseline"], "flow_baseline_logprob", baseline_pred)
    results["blend_scans"] = {
        "flow_seed45_probability": flow_seed_prob,
        "flow_seed45_logprob": flow_seed_log,
        "flow_baseline_probability": flow_base_prob,
        "flow_baseline_logprob": flow_base_log,
    }
    results["best_blends"] = {
        key: max(rows, key=lambda r: (r["macro_f1"], r["accuracy"]))
        for key, rows in results["blend_scans"].items()
    }

    flow_score = logp(artifacts["flow"])
    flow_strict_probs, flow_strict_folds = apply_strict_bias(y, flow_score, fold_ids)
    flow_full_probs, flow_full_bias = full_bias(y, flow_score)
    np.save("oof/fields_flow_ep3_strict_bias_probs.npy", flow_strict_probs.astype(np.float32))
    np.save("oof/fields_flow_ep3_full_oof_bias_probs.npy", flow_full_probs.astype(np.float32))
    results["bias"] = {
        "flow_strict_crossfit": {
            "metrics": evaluate("flow_strict_crossfit_bias", y, flow_strict_probs, baseline_pred),
            "folds": flow_strict_folds,
        },
        "flow_full_oof_fit_optimistic": {
            "metrics": evaluate("flow_full_oof_bias_optimistic", y, flow_full_probs, baseline_pred),
            "bias": flow_full_bias,
        },
    }

    greedy_probs, greedy_info = greedy_forward(y, artifacts, baseline_pred)
    np.save("oof/current_greedy_flow_seed45_baseline_probs.npy", greedy_probs.astype(np.float32))
    results["greedy_forward"] = {
        "metrics": evaluate("greedy_forward_prob_logprob", y, greedy_probs, baseline_pred, seed45_pred),
        **greedy_info,
    }
    greedy_strict_probs, greedy_strict_folds = apply_strict_bias(y, logp(greedy_probs), fold_ids)
    np.save("oof/current_greedy_flow_seed45_baseline_strict_bias_probs.npy", greedy_strict_probs.astype(np.float32))
    results["best_candidate_strict_bias"] = {
        "base_candidate": "greedy_forward",
        "metrics": evaluate("greedy_forward_strict_crossfit_bias", y, greedy_strict_probs, baseline_pred, seed45_pred),
        "folds": greedy_strict_folds,
    }

    best_name = "greedy_forward_strict_crossfit_bias"
    best_probs = greedy_strict_probs
    best_pred = best_probs.argmax(axis=1)
    results["analysis"] = {
        "paired_baseline_flow": paired(y, baseline_pred, preds["flow"], "baseline", "flow"),
        "paired_baseline_seed45": paired(y, baseline_pred, preds["seed45"], "baseline", "seed45"),
        "paired_baseline_best": paired(y, baseline_pred, best_pred, "baseline", best_name),
        "per_class_best": per_class_report(y, baseline_pred, best_pred),
        "per_class_flow": per_class_report(y, baseline_pred, preds["flow"]),
    }

    Path(os.path.dirname(args.out_json)).mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows = []
    for row in results["singles"]:
        rows.append(row)
    rows.extend(results["best_blends"].values())
    rows.append(results["bias"]["flow_strict_crossfit"]["metrics"])
    rows.append(results["bias"]["flow_full_oof_fit_optimistic"]["metrics"])
    rows.append(results["greedy_forward"]["metrics"])
    rows.append(results["best_candidate_strict_bias"]["metrics"])
    rows_sorted = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)

    md = []
    md.append("# Current OOF Ensemble Analysis\n")
    md.append("## Validation\n")
    md.append(f"- N: {n}\n")
    md.append(f"- Class order: `{CLASSES}`\n")
    md.append(f"- Fold counts: `{results['validation']['fold_counts']}`\n")
    md.append("- Seed45 raw logits were used only for inventory validation; FLOW has no raw logits, so log-space blends use clipped log probabilities.\n")
    md.append("\n## Ranked Candidates\n")
    md.append("| candidate | macro_f1 | accuracy | delta_macro_vs_baseline | net_correct_gain |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in rows_sorted:
        md.append(
            f"| {r['name']} | {r['macro_f1']:.6f} | {r['accuracy']:.6f} | "
            f"{r.get('delta_macro_vs_baseline', 0.0):+.6f} | {r.get('net_correct_gain_vs_baseline', 0)} |\n"
        )
    md.append("\n## Best Blend Details\n")
    for k, r in results["best_blends"].items():
        md.append(f"- {k}: weight_b={r['weight_b']:.2f}, macro_f1={r['macro_f1']:.6f}, acc={r['accuracy']:.6f}\n")
    md.append("\n## Greedy Trace\n")
    for item in results["greedy_forward"]["trace"]:
        md.append(f"- `{item}`\n")
    md.append("\n## Best Candidate Paired vs Baseline\n")
    md.append(f"`{results['analysis']['paired_baseline_best']}`\n")
    md.append("\n## Per-Class Delta for Best Candidate\n")
    md.append("| class | support | baseline_f1 | candidate_f1 | delta |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for r in sorted(results["analysis"]["per_class_best"], key=lambda x: x["delta_f1"]):
        md.append(f"| {r['class']} | {r['support']} | {r['baseline_f1']:.6f} | {r['candidate_f1']:.6f} | {r['delta_f1']:+.6f} |\n")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("".join(md))

    print(f"saved {args.out_json}")
    print(f"saved {args.out_md}")
    print("TOP")
    for r in rows_sorted[:12]:
        print(r)


if __name__ == "__main__":
    main()
