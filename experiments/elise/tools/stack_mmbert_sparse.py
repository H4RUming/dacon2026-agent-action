import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import CLASSES, CLASS2ID, NUM_CLASSES, load_train, read_jsonl


MMBERT_OOF = "oof/fields_mmbert_ep3.npy"
SPARSE_OOF = "oof/sparse_tfidf_lr.npy"
CANDIDATES = [0.01, 0.03, 0.1, 0.3, 1.0]
CONTEXT_CAT_NAMES = ["last_action", "prev2_action", "prev3_action", "turn_bucket"]
BASE_EXPECTED_F1 = 0.770729
BASE_EXPECTED_ACC = 0.769086


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
        raise RuntimeError("alignment uncertainty: load_train ids differ from train.jsonl row order")

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
        raise RuntimeError("alignment uncertainty: labels do not match id-joined labels")
    return samples, ids, y, groups


def validate_prob(path: str, n: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    p = np.load(path)
    if p.shape != (n, NUM_CLASSES):
        raise ValueError(f"{path}: expected {(n, NUM_CLASSES)}, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"{path}: non-finite values present")
    if p.min() < -1e-6:
        raise ValueError(f"{path}: negative probabilities present")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(f"{path}: probability rows do not sum to 1")
    return p.astype(np.float32, copy=False)


def entropy(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-12, 1.0)
    return -(pc * np.log(pc)).sum(axis=1)


def margin(p: np.ndarray) -> np.ndarray:
    part = np.partition(p, -2, axis=1)
    return part[:, -1] - part[:, -2]


def context_cats(samples: list[dict]) -> np.ndarray:
    rows = []
    for sample in samples:
        meta = sample.get("session_meta", {}) or {}
        a1, a2, a3 = history_actions(sample)
        rows.append([a1, a2, a3, turn_bucket(meta.get("turn_index"))])
    return np.array(rows, dtype=object)


def build_feature_sets(samples: list[dict], p_m: np.ndarray, p_s: np.ndarray):
    logp_m = np.log(np.clip(p_m, 1e-7, 1.0))
    logp_s = np.log(np.clip(p_s, 1e-7, 1.0))
    pred_m = p_m.argmax(axis=1)
    pred_s = p_s.argmax(axis=1)
    ent_m = entropy(p_m)
    ent_s = entropy(p_s)
    conf_m = p_m.max(axis=1)
    conf_s = p_s.max(axis=1)
    mar_m = margin(p_m)
    mar_s = margin(p_s)
    agree = pred_m == pred_s

    m1_num = np.hstack([logp_m, logp_s]).astype(np.float32)
    conf_num = np.column_stack([ent_m, conf_m, mar_m, ent_s, conf_s, mar_s]).astype(np.float32)
    m2_num = np.hstack([m1_num, conf_num]).astype(np.float32)
    m2_cat = np.column_stack([
        np.where(agree, "agree", "disagree"),
        np.array([CLASSES[i] for i in pred_m], dtype=object),
        np.array([CLASSES[i] for i in pred_s], dtype=object),
    ]).astype(object)
    ctx_cat = context_cats(samples)
    m3_cat = np.hstack([m2_cat, ctx_cat]).astype(object)
    gate_num = np.hstack([
        m2_num,
        (conf_s - conf_m)[:, None],
        (ent_s - ent_m)[:, None],
        (mar_s - mar_m)[:, None],
    ]).astype(np.float32)
    gate_cat = np.hstack([m2_cat[:, 1:3], ctx_cat]).astype(object)
    return {
        "m1": (m1_num, None),
        "m2": (m2_num, m2_cat),
        "m3": (m2_num, m3_cat),
        "gate": (gate_num, gate_cat),
        "pred_m": pred_m,
        "pred_s": pred_s,
        "agree": agree,
        "disagree": ~agree,
    }


def fit_encoder(numeric: np.ndarray, cats: np.ndarray | None, train_idx: np.ndarray):
    mean = numeric[train_idx].mean(axis=0)
    std = numeric[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    vocabs = []
    if cats is not None and cats.shape[1] > 0:
        for j in range(cats.shape[1]):
            vals = sorted(set(str(v) for v in cats[train_idx, j]))
            vocabs.append({v: i for i, v in enumerate(vals)})
    offsets = []
    cur = numeric.shape[1]
    for vocab in vocabs:
        offsets.append(cur)
        cur += len(vocab)
    return mean.astype(np.float32), std.astype(np.float32), vocabs, offsets, cur


def transform_features(numeric: np.ndarray, cats: np.ndarray | None, idx: np.ndarray, encoder) -> np.ndarray:
    mean, std, vocabs, offsets, dim = encoder
    x = np.zeros((len(idx), dim), dtype=np.float32)
    x[:, : numeric.shape[1]] = (numeric[idx] - mean) / std
    if cats is not None:
        for col, vocab in enumerate(vocabs):
            off = offsets[col]
            for row_pos, raw in enumerate(cats[idx, col]):
                mapped = vocab.get(str(raw))
                if mapped is not None:
                    x[row_pos, off + mapped] = 1.0
    return x


def fit_predict_linear(
    numeric: np.ndarray,
    cats: np.ndarray | None,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    c: float,
    device: torch.device,
    n_classes: int,
    seed: int,
) -> np.ndarray:
    if len(np.unique(y[train_idx])) < 2:
        raise RuntimeError("train split has fewer than 2 classes")
    encoder = fit_encoder(numeric, cats, train_idx)
    x_tr_np = transform_features(numeric, cats, train_idx, encoder)
    x_va_np = transform_features(numeric, cats, val_idx, encoder)
    x_tr = torch.from_numpy(x_tr_np).to(device)
    y_tr = torch.from_numpy(y[train_idx]).long().to(device)
    x_va = torch.from_numpy(x_va_np).to(device)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = torch.nn.Linear(x_tr.shape[1], n_classes).to(device)
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


def score_for_selection(y_true, probs, n_classes: int) -> float:
    pred = probs.argmax(axis=1)
    if n_classes == NUM_CLASSES:
        return macro(y_true, pred)
    return f1_score(y_true, pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def select_c_nested(numeric, cats, y, groups, train_idx, seed, device, n_classes):
    y_train = y[train_idx]
    groups_train = groups[train_idx]
    counts = Counter(y_train)
    min_class_count = min(counts.values())
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
            probs = fit_predict_linear(numeric, cats, y, inner_tr, inner_va, c, device, n_classes, seed)
            fold_scores.append(score_for_selection(y[inner_va], probs, n_classes))
        scores[c] = float(np.mean(fold_scores))
    best_c = max(CANDIDATES, key=lambda c: (scores[c], -c))
    return best_c, scores


def crossfit_multiclass(name, numeric, cats, y, groups, device, train_mask=None, predict_mask=None, base_probs=None):
    out = np.zeros((len(y), NUM_CLASSES), dtype=np.float32) if base_probs is None else base_probs.copy()
    selected, fold_info, warnings = [], [], []
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        tr_fit = tr if train_mask is None else tr[train_mask[tr]]
        va_pred = va if predict_mask is None else va[predict_mask[va]]
        t0 = time.time()
        if len(va_pred) == 0:
            selected.append(None)
            fold_info.append({"fold": fold, "selected_c": None, "inner_scores": {}, "fit_time_s": 0.0})
            continue
        if len(tr_fit) < 100 or len(np.unique(y[tr_fit])) < NUM_CLASSES:
            msg = f"{name} fold {fold}: degenerate train subset; keeping base probabilities"
            log("WARNING: " + msg)
            warnings.append(msg)
            selected.append(None)
            fold_info.append({"fold": fold, "selected_c": None, "inner_scores": {}, "fit_time_s": 0.0})
            continue
        c, scores = select_c_nested(numeric, cats, y, groups, tr_fit, 4242 + fold, device, NUM_CLASSES)
        probs = fit_predict_linear(numeric, cats, y, tr_fit, va_pred, c, device, NUM_CLASSES, 9000 + fold)
        out[va_pred] = probs
        elapsed = time.time() - t0
        log(f"{name} fold {fold}: selected_C={c} inner={scores} train={len(tr_fit)} val={len(va_pred)} fit_time_s={elapsed:.1f}")
        selected.append(c)
        fold_info.append({"fold": fold, "selected_c": c, "inner_scores": scores, "fit_time_s": elapsed})
    return out, selected, fold_info, warnings


def crossfit_binary_gate(numeric, cats, y, groups, p_m, p_s, pred_m, pred_s, disagree, device):
    out = p_m.copy()
    q_sparse = np.zeros(len(y), dtype=np.float32)
    selected, fold_info, warnings = [], [], []
    exclusive = disagree & ((pred_m == y) ^ (pred_s == y))
    gate_y = np.zeros(len(y), dtype=np.int64)
    gate_y[exclusive & (pred_s == y)] = 1
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        tr_gate = tr[exclusive[tr]]
        va_dis = va[disagree[va]]
        t0 = time.time()
        if len(va_dis) == 0:
            selected.append(None)
            fold_info.append({"fold": fold, "selected_c": None, "inner_scores": {}, "fit_time_s": 0.0})
            continue
        if len(tr_gate) < 100 or len(np.unique(gate_y[tr_gate])) < 2:
            msg = f"binary_gate fold {fold}: too few exclusive-correct rows; keeping mmBERT"
            log("WARNING: " + msg)
            warnings.append(msg)
            selected.append(None)
            fold_info.append({"fold": fold, "selected_c": None, "inner_scores": {}, "fit_time_s": 0.0})
            continue
        c, scores = select_c_nested(numeric, cats, gate_y, groups, tr_gate, 6242 + fold, device, 2)
        probs2 = fit_predict_linear(numeric, cats, gate_y, tr_gate, va_dis, c, device, 2, 12000 + fold)
        q_sparse[va_dis] = probs2[:, 1]
        choose_sparse = probs2.argmax(axis=1) == 1
        out[va_dis[choose_sparse]] = p_s[va_dis[choose_sparse]]
        elapsed = time.time() - t0
        log(
            f"binary_gate fold {fold}: selected_C={c} inner={scores} "
            f"train_exclusive={len(tr_gate)} val_disagree={len(va_dis)} fit_time_s={elapsed:.1f}"
        )
        selected.append(c)
        fold_info.append({"fold": fold, "selected_c": c, "inner_scores": scores, "fit_time_s": elapsed})
    return out, q_sparse, selected, fold_info, warnings


def threshold_gate_probs(p_m, p_s, pred_m, pred_s, disagree, q_sparse, threshold):
    out = p_m.copy()
    switch = disagree & (q_sparse >= threshold)
    out[switch] = p_s[switch]
    return out


def save_probs(path: str, arr: np.ndarray):
    if arr.shape != (arr.shape[0], NUM_CLASSES):
        raise ValueError(f"{path}: bad probability shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: non-finite probabilities")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr.astype(np.float32))
    log(f"saved {path} shape={arr.shape} dtype=float32")


def evaluate(name, y, probs, base_pred, pred_s=None, disagree=None):
    pred = probs.argmax(axis=1)
    base_correct = base_pred == y
    meta_correct = pred == y
    row = {
        "experiment": name,
        "macro_f1": macro(y, pred),
        "accuracy": float(accuracy_score(y, pred)),
        "delta_vs_mmbert": macro(y, pred) - macro(y, base_pred),
        "changed_prediction_count": int((pred != base_pred).sum()),
        "mmbert_wrong_to_meta_correct": int((~base_correct & meta_correct).sum()),
        "mmbert_correct_to_meta_wrong": int((base_correct & ~meta_correct).sum()),
    }
    row["net_correct_gain"] = row["mmbert_wrong_to_meta_correct"] - row["mmbert_correct_to_meta_wrong"]
    if pred_s is not None and disagree is not None:
        switch_sparse = disagree & (pred == pred_s) & (pred != base_pred)
        denom = int(switch_sparse.sum())
        sparse_only = switch_sparse & (pred_s == y) & (base_pred != y)
        row.update(
            {
                "disagreement_accuracy": float((pred[disagree] == y[disagree]).mean()) if disagree.any() else None,
                "sparse_only_correct_rescue_count": int(sparse_only.sum()),
                "switches_to_sparse": denom,
                "sparse_switch_precision": float(sparse_only.sum() / denom) if denom else None,
            }
        )
    return row


def oracle_probs(y, p_m, p_s):
    pred_m = p_m.argmax(axis=1)
    pred_s = p_s.argmax(axis=1)
    out_pred = pred_m.copy()
    sparse_correct = pred_s == y
    mm_wrong = pred_m != y
    out_pred[mm_wrong & sparse_correct] = pred_s[mm_wrong & sparse_correct]
    out = np.zeros_like(p_m)
    out[np.arange(len(y)), out_pred] = 1.0
    return out


def print_per_class(name, y, base_pred, meta_pred):
    base_f1 = precision_recall_fscore_support(y, base_pred, labels=list(range(NUM_CLASSES)), zero_division=0)[2]
    meta_f1 = precision_recall_fscore_support(y, meta_pred, labels=list(range(NUM_CLASSES)), zero_division=0)[2]
    support = np.bincount(y, minlength=NUM_CLASSES)
    log(f"\nPER_CLASS {name}")
    for i, cls in enumerate(CLASSES):
        log(
            f"  {cls:16s} support={int(support[i]):5d} "
            f"mmBERT_f1={base_f1[i]:.6f} meta_f1={meta_f1[i]:.6f} delta={meta_f1[i] - base_f1[i]:+.6f}"
        )


def print_rescue_concentration(y, pred_m, pred_s, disagree):
    log("\nSPARSE_RESCUE_CONCENTRATION")
    rows = []
    for i, cls in enumerate(CLASSES):
        mask = y == i
        sparse_only = int(((pred_s == y) & (pred_m != y) & mask).sum())
        mmbert_only = int(((pred_m == y) & (pred_s != y) & mask).sum())
        dis = int((disagree & mask).sum())
        rows.append((sparse_only, cls, int(mask.sum()), mmbert_only, dis))
    for sparse_only, cls, support, mmbert_only, dis in sorted(rows, reverse=True):
        log(
            f"  {cls:16s} support={support:5d} sparse_only_correct={sparse_only:4d} "
            f"mmbert_only_correct={mmbert_only:4d} disagreement={dis:4d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/original_data")
    parser.add_argument("--mmbert_oof", default=MMBERT_OOF)
    parser.add_argument("--sparse_oof", default=SPARSE_OOF)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(requested_device)
    log(f"device {device}")
    if device.type == "cuda":
        log(f"gpu {torch.cuda.get_device_name(0)}")

    samples, ids, y, groups = load_labels_and_samples(args.data_dir)
    p_m = validate_prob(args.mmbert_oof, len(y))
    p_s = validate_prob(args.sparse_oof, len(y))
    if p_m.shape != p_s.shape:
        raise RuntimeError("OOF shape mismatch")

    log("alignment_method train.jsonl row order via src.data.load_train; labels joined by id; groups=session id")
    log("fold_split_method StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42); inner seed=4242+outer_fold")
    log(f"N {len(y)}")
    log(f"mmBERT_shape {p_m.shape}")
    log(f"sparse_shape {p_s.shape}")
    log(f"label_count {len(y)}")
    log(f"group_count {len(set(groups))}")
    log(f"class_order {CLASSES}")
    if len(y) != 70000 or NUM_CLASSES != 14:
        raise RuntimeError(f"unexpected data shape/classes: n={len(y)} classes={NUM_CLASSES}")

    feats = build_feature_sets(samples, p_m, p_s)
    pred_m = feats["pred_m"]
    pred_s = feats["pred_s"]
    disagree = feats["disagree"]
    base_f1 = macro(y, pred_m)
    base_acc = float((pred_m == y).mean())
    log(f"M0 mmbert_raw macro_f1={base_f1:.6f} accuracy={base_acc:.6f}")
    if abs(base_f1 - BASE_EXPECTED_F1) > 0.001 or abs(base_acc - BASE_EXPECTED_ACC) > 0.001:
        raise RuntimeError(
            f"alignment error suspected: mmBERT metrics {base_f1:.6f}/{base_acc:.6f} "
            f"materially differ from expected {BASE_EXPECTED_F1:.6f}/{BASE_EXPECTED_ACC:.6f}"
        )

    both_correct = (pred_m == y) & (pred_s == y)
    mm_only = (pred_m == y) & (pred_s != y)
    sparse_only = (pred_m != y) & (pred_s == y)
    both_wrong = (pred_m != y) & (pred_s != y)
    oracle = oracle_probs(y, p_m, p_s)
    log("\nORACLE_ANALYSIS")
    log(f"disagreement_ratio {float(disagree.mean()):.6f}")
    log(f"both_correct {int(both_correct.sum())} ratio={both_correct.mean():.6f}")
    log(f"mmbert_only_correct {int(mm_only.sum())} ratio={mm_only.mean():.6f}")
    log(f"sparse_only_correct {int(sparse_only.sum())} ratio={sparse_only.mean():.6f}")
    log(f"both_wrong {int(both_wrong.sum())} ratio={both_wrong.mean():.6f}")
    log(f"top1_oracle_accuracy {float((oracle.argmax(axis=1) == y).mean()):.6f}")
    log(f"oracle_macro_f1_upper_bound {macro(y, oracle.argmax(axis=1)):.6f}")
    if abs(disagree.mean() - 0.370786) > 0.002 or abs(sparse_only.mean() - 0.035) > 0.002:
        raise RuntimeError("oracle/diversity mismatch; stopping on possible alignment issue")

    results = [evaluate("mmbert_raw", y, p_m, pred_m)]

    m1_probs, m1_sel, m1_folds, m1_warn = crossfit_multiclass("minimal_28d", *feats["m1"], y, groups, device)
    save_probs("oof/meta_mmbert_sparse_minimal.npy", m1_probs)
    results.append(evaluate("minimal_28d", y, m1_probs, pred_m))

    m2_probs, m2_sel, m2_folds, m2_warn = crossfit_multiclass("confidence", *feats["m2"], y, groups, device)
    save_probs("oof/meta_mmbert_sparse_confidence.npy", m2_probs)
    results.append(evaluate("confidence", y, m2_probs, pred_m))

    m3_probs, m3_sel, m3_folds, m3_warn = crossfit_multiclass("contextual", *feats["m3"], y, groups, device)
    save_probs("oof/meta_mmbert_sparse_context.npy", m3_probs)
    results.append(evaluate("contextual", y, m3_probs, pred_m))

    m4_probs, m4_sel, m4_folds, m4_warn = crossfit_multiclass(
        "disagree_14way", *feats["m3"], y, groups, device, train_mask=disagree, predict_mask=disagree, base_probs=p_m
    )
    save_probs("oof/meta_mmbert_sparse_disagree14.npy", m4_probs)
    results.append(evaluate("disagree_14way", y, m4_probs, pred_m, pred_s, disagree))

    m5_probs, q_sparse, m5_sel, m5_folds, m5_warn = crossfit_binary_gate(
        *feats["gate"], y, groups, p_m, p_s, pred_m, pred_s, disagree, device
    )
    save_probs("oof/meta_mmbert_sparse_binary_gate.npy", m5_probs)
    results.append(evaluate("binary_gate_0.50", y, m5_probs, pred_m, pred_s, disagree))
    threshold_rows = []
    for threshold in [0.60, 0.70, 0.80, 0.90]:
        probs = threshold_gate_probs(p_m, p_s, pred_m, pred_s, disagree, q_sparse, threshold)
        row = evaluate(f"binary_gate_{threshold:.2f}", y, probs, pred_m, pred_s, disagree)
        results.append(row)
        threshold_rows.append(row)

    oracle_row = evaluate("oracle_upper_bound", y, oracle, pred_m)
    results.append(oracle_row)

    print_rescue_concentration(y, pred_m, pred_s, disagree)
    for name, probs in [
        ("minimal_28d", m1_probs),
        ("confidence", m2_probs),
        ("contextual", m3_probs),
        ("disagree_14way", m4_probs),
        ("binary_gate_0.50", m5_probs),
    ]:
        print_per_class(name, y, pred_m, probs.argmax(axis=1))

    all_warnings = m1_warn + m2_warn + m3_warn + m4_warn + m5_warn
    log("\nSELECTED_C")
    log(f"minimal_28d {m1_sel}")
    log(f"confidence {m2_sel}")
    log(f"contextual {m3_sel}")
    log(f"disagree_14way {m4_sel}")
    log(f"binary_gate {m5_sel}")

    log("\nGATE_SWITCH_PRECISION")
    for row in results:
        if row["experiment"].startswith("binary_gate") or row["experiment"] == "disagree_14way":
            log(
                f"{row['experiment']} switches={row.get('switches_to_sparse')} "
                f"precision={row.get('sparse_switch_precision')} "
                f"rescue={row.get('sparse_only_correct_rescue_count')} "
                f"disagree_acc={row.get('disagreement_accuracy')}"
            )

    log("\nFINAL_RESULT_TABLE")
    log("experiment | macro_f1 | accuracy | delta_vs_mmbert")
    for row in results:
        log(f"{row['experiment']} | {row['macro_f1']:.6f} | {row['accuracy']:.6f} | {row['delta_vs_mmbert']:+.6f}")

    best_meta = max((r for r in results if r["experiment"] != "oracle_upper_bound"), key=lambda r: r["delta_vs_mmbert"])
    gain = best_meta["delta_vs_mmbert"]
    if gain <= 0.001:
        decision = "sparse diversity not practically recoverable by linear meta gating; stop this direction"
    elif gain < 0.003:
        decision = "weak signal; consider only if implementation cost is negligible"
    elif gain < 0.005:
        decision = "strong complementary branch; validate robustness and submission integration"
    else:
        decision = "major lever; prioritize immediately"
    log(f"decision {decision}")
    if any(r.get("sparse_switch_precision") is not None and r["sparse_switch_precision"] < 0.55 for r in results):
        log("routing_note switch precision below 0.55 indicates gate is not learning meaningful routing")
    log("warning sparse TF-IDF LR OOF used fixed C=1.0 max_iter=20 with convergence warnings; probability calibration may be poor")
    if all_warnings:
        log("warnings:")
        for warning in all_warnings:
            log(f"  {warning}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "class_order": CLASSES,
        "selected_c": {
            "minimal_28d": m1_sel,
            "confidence": m2_sel,
            "contextual": m3_sel,
            "disagree_14way": m4_sel,
            "binary_gate": m5_sel,
        },
        "folds": {
            "minimal_28d": m1_folds,
            "confidence": m2_folds,
            "contextual": m3_folds,
            "disagree_14way": m4_folds,
            "binary_gate": m5_folds,
        },
        "results": results,
        "oracle": {
            "disagreement_ratio": float(disagree.mean()),
            "sparse_only_correct": int(sparse_only.sum()),
            "mmbert_only_correct": int(mm_only.sum()),
            "both_correct": int(both_correct.sum()),
            "both_wrong": int(both_wrong.sum()),
            "top1_oracle_accuracy": float((oracle.argmax(axis=1) == y).mean()),
            "oracle_macro_f1_upper_bound": macro(y, oracle.argmax(axis=1)),
        },
        "decision": decision,
        "warnings": all_warnings + [
            "sparse TF-IDF LR OOF used fixed C=1.0 max_iter=20 with convergence warnings; probability calibration may be poor"
        ],
    }
    os.makedirs("results", exist_ok=True)
    with open("results/stack_mmbert_sparse_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("saved results/stack_mmbert_sparse_summary.json")
    log(f"finished_at {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
