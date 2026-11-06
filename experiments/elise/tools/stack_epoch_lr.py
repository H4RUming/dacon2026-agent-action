import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import CLASSES, CLASS2ID, NUM_CLASSES, load_train, read_jsonl


CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0]
OOF_PATHS = [
    "oof/fields_mmbert_ep1.npy",
    "oof/fields_mmbert_ep2.npy",
    "oof/fields_mmbert_ep3.npy",
    "oof/fields_mmbert_ep4.npy",
]
CAT_NAMES = ["last_action", "prev2_action", "prev3_action", "turn_bucket"]


def log(msg: str = ""):
    print(msg, flush=True)


def turn_bucket(turn) -> str:
    try:
        t = int(turn)
    except (TypeError, ValueError):
        return "__MISSING__"
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


def history_actions(sample: dict) -> tuple[str, str, str]:
    actions = [
        h.get("name") or "__NONE__"
        for h in (sample.get("history", []) or [])
        if h.get("role") == "assistant_action"
    ]
    recent = list(reversed(actions))
    return tuple(recent[i] if i < len(recent) else "__NONE__" for i in range(3))


def load_labels_and_samples(data_dir: str):
    samples = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    ids, _, y, groups = load_train(data_dir)
    ids = np.array(ids)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups)

    json_ids = np.array([s["id"] for s in samples])
    if len(samples) != len(ids) or not np.array_equal(ids, json_ids):
        raise RuntimeError("alignment uncertainty: load_train ids differ from train.jsonl order")

    label_ids, label_actions = [], []
    with open(os.path.join(data_dir, "train_labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label_ids.append(row["id"])
            label_actions.append(row["action"])
    if set(label_ids) != set(ids):
        raise RuntimeError("alignment uncertainty: train_labels.csv ids differ from train.jsonl ids")
    label_map = dict(zip(label_ids, label_actions))
    y_from_map = np.array([CLASS2ID[label_map[sid]] for sid in ids], dtype=np.int64)
    if not np.array_equal(y, y_from_map):
        raise RuntimeError("alignment uncertainty: labels do not match load_train join by id")
    return samples, ids, y, groups


def load_oof(paths: list[str], n_expected: int) -> tuple[list[np.ndarray], str]:
    arrays, kind = [], None
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        arr = np.load(path)
        if arr.shape != (n_expected, NUM_CLASSES):
            raise ValueError(f"{path}: expected {(n_expected, NUM_CLASSES)}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{path}: non-finite values present")
        row_sum = arr.sum(axis=1)
        looks_prob = (
            arr.min() >= -1e-6
            and arr.max() <= 1.0 + 1e-5
            and np.allclose(row_sum, 1.0, atol=1e-3)
        )
        this_kind = "probability" if looks_prob else "logit"
        if kind is None:
            kind = this_kind
        elif kind != this_kind:
            raise ValueError(f"mixed OOF array kinds: first={kind}, {path}={this_kind}")
        arrays.append(arr.astype(np.float64, copy=False))
    return arrays, kind or "unknown"


def entropy(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-12, 1.0)
    return -(pc * np.log(pc)).sum(axis=1)


def margins(p: np.ndarray) -> np.ndarray:
    part = np.partition(p, -2, axis=1)
    return part[:, -1] - part[:, -2]


def build_features(samples: list[dict], arrays: list[np.ndarray], kind: str):
    if kind == "probability":
        probs = arrays
        logps = [np.log(np.clip(p, 1e-7, 1.0)) for p in probs]
    else:
        logits = arrays
        shifted = [z - z.max(axis=1, keepdims=True) for z in logits]
        exps = [np.exp(z) for z in shifted]
        probs = [e / e.sum(axis=1, keepdims=True) for e in exps]
        logps = [np.log(np.clip(p, 1e-7, 1.0)) for p in probs]

    numeric_parts = []
    numeric_names = []
    for ep, lp in enumerate(logps, start=1):
        numeric_parts.append(lp)
        numeric_names.extend([f"ep{ep}_logp_{cls}" for cls in CLASSES])
    ent = np.stack([entropy(p) for p in probs], axis=1)
    mar = np.stack([margins(p) for p in probs], axis=1)
    numeric_parts.extend([ent, mar])
    numeric_names.extend([f"entropy_ep{i}" for i in range(1, 5)])
    numeric_names.extend([f"margin_ep{i}" for i in range(1, 5)])

    turn_indices, cat_rows = [], []
    for sample in samples:
        meta = sample.get("session_meta", {}) or {}
        turn = meta.get("turn_index")
        try:
            turn_indices.append(float(turn))
        except (TypeError, ValueError):
            turn_indices.append(-1.0)
        a1, a2, a3 = history_actions(sample)
        cat_rows.append([a1, a2, a3, turn_bucket(turn)])
    numeric_parts.append(np.array(turn_indices, dtype=np.float64)[:, None])
    numeric_names.append("turn_index")
    numeric = np.concatenate(numeric_parts, axis=1).astype(np.float32)
    cats = np.array(cat_rows, dtype=object)
    return numeric, cats, numeric_names, CAT_NAMES, probs


def fit_encoder(numeric: np.ndarray, cats: np.ndarray, train_idx: np.ndarray):
    mean = numeric[train_idx].mean(axis=0)
    std = numeric[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    vocabs = []
    for j in range(cats.shape[1]):
        vals = sorted(set(str(v) for v in cats[train_idx, j]))
        vocabs.append({v: i for i, v in enumerate(vals)})
    offsets = []
    cur = numeric.shape[1]
    for vocab in vocabs:
        offsets.append(cur)
        cur += len(vocab)
    return mean.astype(np.float32), std.astype(np.float32), vocabs, offsets, cur


def transform_features(numeric: np.ndarray, cats: np.ndarray, idx: np.ndarray, encoder):
    mean, std, vocabs, offsets, dim = encoder
    x = np.zeros((len(idx), dim), dtype=np.float32)
    x[:, : numeric.shape[1]] = (numeric[idx] - mean) / std
    for col, vocab in enumerate(vocabs):
        off = offsets[col]
        for row_pos, raw in enumerate(cats[idx, col]):
            mapped = vocab.get(str(raw))
            if mapped is not None:
                x[row_pos, off + mapped] = 1.0
    return x


def fit_predict_gpu(
    numeric: np.ndarray,
    cats: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    c: float,
    device: torch.device,
) -> np.ndarray:
    if len(np.unique(y[train_idx])) < 2:
        raise RuntimeError("train split has fewer than 2 classes")
    encoder = fit_encoder(numeric, cats, train_idx)
    x_tr_np = transform_features(numeric, cats, train_idx, encoder)
    x_va_np = transform_features(numeric, cats, val_idx, encoder)
    x_tr = torch.from_numpy(x_tr_np).to(device)
    y_tr = torch.from_numpy(y[train_idx]).long().to(device)
    x_va = torch.from_numpy(x_va_np).to(device)

    torch.manual_seed(42)
    model = torch.nn.Linear(x_tr.shape[1], NUM_CLASSES).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=80,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-5,
        tolerance_change=1e-7,
    )
    reg = 1.0 / c
    n_train = float(len(train_idx))

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tr)
        l2 = model.weight.pow(2).sum()
        loss = loss_fn(logits, y_tr) + (0.5 * reg / n_train) * l2
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        probs = torch.softmax(model(x_va), dim=1).detach().cpu().numpy()
    del model, x_tr, y_tr, x_va
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def macro(y_true, pred) -> float:
    return f1_score(y_true, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)


def evaluate_probs(name: str, y: np.ndarray, probs: np.ndarray, ep3_pred: np.ndarray):
    pred = probs.argmax(axis=1)
    return {
        "experiment": name,
        "macro_f1": macro(y, pred),
        "accuracy": float((pred == y).mean()),
        "delta_vs_ep3": macro(y, pred) - macro(y, ep3_pred),
    }


def select_c_nested(numeric, cats, y, groups, train_idx, seed, device):
    y_train = y[train_idx]
    groups_train = groups[train_idx]
    min_class_count = min(Counter(y_train).values())
    n_splits = min(3, min_class_count)
    if n_splits < 2:
        raise RuntimeError("inner CV impossible: fewer than 2 examples in at least one class")
    inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = {}
    for c in CANDIDATES:
        fold_scores = []
        for inner_tr_rel, inner_va_rel in inner.split(np.zeros(len(train_idx)), y_train, groups_train):
            inner_tr = train_idx[inner_tr_rel]
            inner_va = train_idx[inner_va_rel]
            probs = fit_predict_gpu(numeric, cats, y, inner_tr, inner_va, c, device)
            fold_scores.append(macro(y[inner_va], probs.argmax(axis=1)))
        scores[c] = float(np.mean(fold_scores))
    best_c = max(CANDIDATES, key=lambda c: (scores[c], -c))
    return best_c, scores


def crossfit_full(numeric, cats, y, groups, device):
    out = np.zeros((len(y), NUM_CLASSES), dtype=np.float64)
    selected = []
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        c, scores = select_c_nested(numeric, cats, y, groups, tr, 4242 + fold, device)
        log(f"E1 fold {fold}: selected_C={c} inner={scores}")
        out[va] = fit_predict_gpu(numeric, cats, y, tr, va, c, device)
        selected.append(c)
    return out, selected


def crossfit_disagree_train(numeric, cats, y, groups, disagree, ep3_probs, device):
    out = ep3_probs.copy()
    selected, warnings = [], []
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        tr_dis = tr[disagree[tr]]
        va_dis = va[disagree[va]]
        if len(va_dis) == 0:
            selected.append(None)
            continue
        if len(tr_dis) < 100 or len(np.unique(y[tr_dis])) < NUM_CLASSES:
            msg = f"E2 fold {fold}: disagreement train degenerate; keeping ep3"
            log("WARNING: " + msg)
            warnings.append(msg)
            selected.append(None)
            continue
        try:
            c, scores = select_c_nested(numeric, cats, y, groups, tr_dis, 5242 + fold, device)
            log(f"E2 fold {fold}: selected_C={c} inner={scores} train_disagree={len(tr_dis)} val_disagree={len(va_dis)}")
            out[va_dis] = fit_predict_gpu(numeric, cats, y, tr_dis, va_dis, c, device)
            selected.append(c)
        except Exception as e:
            msg = f"E2 fold {fold}: {e}; keeping ep3"
            log("WARNING: " + msg)
            warnings.append(msg)
            selected.append(None)
    return out, selected, warnings


def crossfit_disagree_apply(full_probs, disagree, ep3_probs):
    out = ep3_probs.copy()
    out[disagree] = full_probs[disagree]
    return out


def subset_flip_counts(name, y, ep3_pred, meta_pred, mask):
    ep3_wrong_meta_correct = ((ep3_pred[mask] != y[mask]) & (meta_pred[mask] == y[mask])).sum()
    ep3_correct_meta_wrong = ((ep3_pred[mask] == y[mask]) & (meta_pred[mask] != y[mask])).sum()
    return {
        "subset": name,
        "changed": int((meta_pred[mask] != ep3_pred[mask]).sum()),
        "ep3_wrong_to_meta_correct": int(ep3_wrong_meta_correct),
        "ep3_correct_to_meta_wrong": int(ep3_correct_meta_wrong),
        "net_correct_gain": int(ep3_wrong_meta_correct - ep3_correct_meta_wrong),
    }


def print_flip_analysis(exp_name, y, ep3_pred, meta_pred, agree, disagree):
    log(f"\nERROR FLIPS: {exp_name}")
    for row in [
        subset_flip_counts("overall", y, ep3_pred, meta_pred, np.ones(len(y), dtype=bool)),
        subset_flip_counts("agree", y, ep3_pred, meta_pred, agree),
        subset_flip_counts("disagree", y, ep3_pred, meta_pred, disagree),
    ]:
        log(str(row))
    ep3_f1 = precision_recall_fscore_support(y, ep3_pred, labels=list(range(NUM_CLASSES)), zero_division=0)[2]
    meta_f1 = precision_recall_fscore_support(y, meta_pred, labels=list(range(NUM_CLASSES)), zero_division=0)[2]
    support = np.bincount(y, minlength=NUM_CLASSES)
    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append((meta_f1[i] - ep3_f1[i], cls, int(support[i]), ep3_f1[i], meta_f1[i]))
    log("per_class_delta sorted_by_delta:")
    for delta, cls, sup, base, meta in sorted(rows):
        log(f"  {cls:16s} support={sup:5d} ep3_f1={base:.6f} meta_f1={meta:.6f} delta={delta:+.6f}")


def print_diagnostics(y, probs):
    preds = [p.argmax(axis=1) for p in probs]
    agree = np.ones(len(y), dtype=bool)
    for pred in preds[1:]:
        agree &= pred == preds[0]
    disagree = ~agree
    log(f"all_agree_ratio {agree.mean():.6f}")
    log(f"disagree_ratio {disagree.mean():.6f}")
    for i, pred in enumerate(preds, start=1):
        log(f"ep{i}_acc_agree {(pred[agree] == y[agree]).mean():.6f}")
        log(f"ep{i}_acc_disagree {(pred[disagree] == y[disagree]).mean():.6f}")
    oracle_hit = np.zeros(len(y), dtype=bool)
    for pred in preds:
        oracle_hit |= pred == y
    log(f"top1_oracle_acc {oracle_hit.mean():.6f}")
    log("pairwise_epoch_disagreement:")
    for i in range(len(preds)):
        vals = [f"{(preds[i] != preds[j]).mean():.6f}" for j in range(len(preds))]
        log(f"  ep{i + 1}: " + " ".join(vals))
    for i, pred in enumerate(preds):
        others_correct = np.zeros(len(y), dtype=bool)
        for j, pred_j in enumerate(preds):
            if i != j:
                others_correct |= pred_j == y
        unique = (pred == y) & (~others_correct)
        log(f"ep{i + 1}_unique_correct {unique.mean():.6f}")
    return preds, agree, disagree


def oracle_upper_bound(y, preds, ep3_probs):
    preference = [2, 3, 1, 0]
    out_pred = preds[2].copy()
    for i in range(len(y)):
        for ep_idx in preference:
            if preds[ep_idx][i] == y[i]:
                out_pred[i] = y[i]
                break
    out = np.zeros_like(ep3_probs)
    out[np.arange(len(y)), out_pred] = 1.0
    return out


def save_probs(path: str, arr: np.ndarray):
    if arr.shape != (arr.shape[0], NUM_CLASSES):
        raise ValueError(f"{path}: bad probability shape {arr.shape}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr.astype(np.float32))
    log(f"saved {path} shape={arr.shape}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/original_data")
    parser.add_argument("--oof", nargs=4, default=OOF_PATHS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    log(f"device {device}")
    if device.type == "cuda":
        log(f"gpu {torch.cuda.get_device_name(0)}")

    samples, ids, y, groups = load_labels_and_samples(args.data_dir)
    arrays, kind = load_oof(args.oof, len(y))
    numeric, cats, numeric_names, categorical_names, probs = build_features(samples, arrays, kind)

    log("alignment_method train.jsonl order via src.data.load_train; labels joined by id; OOF filled at StratifiedGroupKFold validation indices")
    log(f"n_samples {len(y)}")
    log(f"OOF_shapes {[a.shape for a in arrays]}")
    log(f"OOF_kind {kind}")
    log(f"label_count {len(y)}")
    log(f"group_count {len(set(groups))}")
    log(f"class_order {CLASSES}")
    log(f"numeric_feature_count {len(numeric_names)}")
    log(f"categorical_features {categorical_names}")
    if len(y) != 70000:
        log(f"WARNING: expected 70000 rows, got {len(y)}")
    if NUM_CLASSES != 14:
        raise RuntimeError(f"expected 14 classes, got {NUM_CLASSES}")
    log("WARNING: no reproducible current-run OOF bias tuner found; skipping bias comparison and not applying submit/model/bias.npy")

    preds, agree, disagree = print_diagnostics(y, probs)
    ep3_probs = probs[2]
    ep3_pred = preds[2]
    ep3_score = macro(y, ep3_pred)
    if abs(ep3_score - 0.7707) > 0.005:
        raise RuntimeError(f"alignment error suspected: ep3 macro-F1 {ep3_score:.6f} materially differs from expected 0.7707")

    full_probs, selected_full = crossfit_full(numeric, cats, y, groups, device)
    disagree_train_probs, selected_dis, warnings = crossfit_disagree_train(
        numeric, cats, y, groups, disagree, ep3_probs, device
    )
    disagree_apply_probs = crossfit_disagree_apply(full_probs, disagree, ep3_probs)
    oracle_probs = oracle_upper_bound(y, preds, ep3_probs)

    results = [
        evaluate_probs("ep3_raw", y, ep3_probs, ep3_pred),
        evaluate_probs("lr_full", y, full_probs, ep3_pred),
        evaluate_probs("lr_disagree_train", y, disagree_train_probs, ep3_pred),
        evaluate_probs("lr_disagree_apply", y, disagree_apply_probs, ep3_pred),
        evaluate_probs("oracle_upper_bound", y, oracle_probs, ep3_pred),
    ]

    save_probs("oof/meta_epoch_lr_full.npy", full_probs)
    save_probs("oof/meta_epoch_lr_disagree_train.npy", disagree_train_probs)
    save_probs("oof/meta_epoch_lr_disagree_apply.npy", disagree_apply_probs)

    for name, arr in [
        ("lr_full", full_probs),
        ("lr_disagree_train", disagree_train_probs),
        ("lr_disagree_apply", disagree_apply_probs),
    ]:
        print_flip_analysis(name, y, ep3_pred, arr.argmax(axis=1), agree, disagree)

    log("\nselected_C_full_by_outer_fold " + str(selected_full))
    log("selected_C_disagree_train_by_outer_fold " + str(selected_dis))
    if warnings:
        log("warnings:")
        for w in warnings:
            log(f"  {w}")

    log("\nFINAL_METRIC_TABLE")
    log("experiment | macro_f1 | accuracy | delta_vs_ep3")
    for row in results:
        log(f"{row['experiment']} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['delta_vs_ep3']:+.6f}")

    best_meta = max((r for r in results if r["experiment"].startswith("lr_")), key=lambda r: r["delta_vs_ep3"])
    gain = best_meta["delta_vs_ep3"]
    if gain <= 0.001:
        decision = "epoch diversity not practically recoverable by this LR gate"
    elif gain < 0.005:
        decision = "useful signal; continue contextual stacking"
    elif gain < 0.008:
        decision = "strong lever; prioritize submission integration and richer FLOW features"
    else:
        decision = "major result; stop unrelated GPU sweeps and validate robustness immediately"
    log(f"decision {decision}")
    log(f"finished_at {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
