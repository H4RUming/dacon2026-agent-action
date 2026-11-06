import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import CLASSES, CLASS2ID, NUM_CLASSES, load_train, read_jsonl, serialize_example


MMBERT_OOF = "oof/fields_mmbert_ep3.npy"
SPARSE_OOF = "oof/sparse_tfidf_lr.npy"
CANDIDATES = [0.3, 1.0, 3.0]


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


def count_bucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def history_actions(sample: dict) -> tuple[str, str, str, int]:
    actions = [
        h.get("name") or "__NONE__"
        for h in (sample.get("history", []) or [])
        if h.get("role") == "assistant_action"
    ]
    recent = list(reversed(actions))
    vals = [recent[i] if i < len(recent) else "__NONE__" for i in range(3)]
    return vals[0], vals[1], vals[2], len(actions)


def dominant_code_language(ws: dict) -> str:
    lm = ws.get("language_mix", {}) or {}
    if not lm:
        return "__NONE__"
    return sorted(lm.items(), key=lambda x: (-x[1], x[0]))[0][0]


def struct_features(sample: dict) -> list[str]:
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    last, prev2, prev3, n_actions = history_actions(sample)
    open_files = ws.get("open_files", []) or []
    return [
        last,
        prev2,
        prev3,
        turn_bucket(meta.get("turn_index")),
        str(ws.get("last_ci_status", "__NONE__")),
        str(ws.get("git_dirty", "__NONE__")),
        str(meta.get("language_pref", "__NONE__")),
        str(meta.get("user_tier", "__NONE__")),
        str(int(bool(open_files))),
        count_bucket(len(open_files)),
        dominant_code_language(ws),
        count_bucket(n_actions),
    ]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def load_aligned_data(data_dir: str):
    samples = read_jsonl(os.path.join(data_dir, "train.jsonl"))
    ids, texts_from_loader, y, groups = load_train(data_dir)
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

    texts = [serialize_example(sample) for sample in samples]
    if texts != texts_from_loader:
        raise RuntimeError("alignment uncertainty: serialize_example output differs from load_train texts")
    structs = np.array([struct_features(sample) for sample in samples], dtype=object)
    return samples, ids, texts, structs, y, groups


def load_mmbert(path: str, n: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    p = np.load(path)
    if p.shape != (n, NUM_CLASSES):
        raise ValueError(f"{path}: expected {(n, NUM_CLASSES)}, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"{path}: non-finite values present")
    if p.min() < -1e-6 or p.max() > 1 + 1e-5 or not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(f"{path}: expected probability rows")
    return p.astype(np.float32, copy=False)


def class_dist(y: np.ndarray) -> str:
    counts = np.bincount(y, minlength=NUM_CLASSES)
    return " ".join(f"{CLASSES[i]}={counts[i]}" for i in range(NUM_CLASSES))


def make_word_vectorizer():
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.999,
        sublinear_tf=True,
        strip_accents=None,
        lowercase=False,
        max_features=300_000,
        dtype=np.float32,
    )


def make_char_vectorizer():
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=300_000,
        dtype=np.float32,
    )


def fit_transform_features(texts, structs, tr, va):
    word = make_word_vectorizer()
    char = make_char_vectorizer()
    ohe = make_ohe()

    tr_text = [texts[i] for i in tr]
    va_text = [texts[i] for i in va]
    xw_tr = word.fit_transform(tr_text)
    xw_va = word.transform(va_text)
    xc_tr = char.fit_transform(tr_text)
    xc_va = char.transform(va_text)
    xs_tr = ohe.fit_transform(structs[tr])
    xs_va = ohe.transform(structs[va])

    x_tr = sp.hstack([xw_tr, xc_tr, xs_tr], format="csr", dtype=np.float32)
    x_va = sp.hstack([xw_va, xc_va, xs_va], format="csr", dtype=np.float32)
    if not sp.isspmatrix_csr(x_tr) or not sp.isspmatrix_csr(x_va):
        raise RuntimeError("combined sparse matrix is not CSR")
    if x_tr.shape[0] != len(tr) or x_va.shape[0] != len(va):
        raise RuntimeError("row count mismatch after transform")
    if not np.isfinite(x_tr.data).all() or not np.isfinite(x_va.data).all():
        raise RuntimeError("non-finite TF-IDF data")
    info = {
        "word_vocab": len(word.vocabulary_),
        "char_vocab": len(char.vocabulary_),
        "struct_dim": xs_tr.shape[1],
        "train_shape": x_tr.shape,
        "val_shape": x_va.shape,
        "train_nnz": int(x_tr.nnz),
        "val_nnz": int(x_va.nnz),
        "dtype": str(x_tr.dtype),
    }
    return x_tr, x_va, info


def make_clf(c: float, seed: int, n_jobs: int, max_iter: int):
    return LogisticRegression(
        C=c,
        solver="saga",
        penalty="l2",
        max_iter=max_iter,
        class_weight=None,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=0,
        tol=1e-3,
    )


def fit_lr(x_tr, y_tr, c: float, seed: int, n_jobs: int, max_iter: int):
    clf = make_clf(c, seed, n_jobs, max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x_tr, y_tr)
    conv = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return clf, conv


def aligned_proba(clf, x, n_rows: int) -> np.ndarray:
    raw = clf.predict_proba(x)
    out = np.zeros((n_rows, NUM_CLASSES), dtype=np.float32)
    for j, cls_id in enumerate(clf.classes_):
        out[:, int(cls_id)] = raw[:, j]
    missing = sorted(set(range(NUM_CLASSES)) - set(int(c) for c in clf.classes_))
    if missing:
        log(f"WARNING: missing classes in fold classifier: {missing}")
    return out


def macro(y, pred) -> float:
    return f1_score(y, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)


def run_inner_selection(texts, structs, y, groups, outer_tr, outer_fold, args):
    if args.selection == "fixed":
        return args.fixed_c, {"fixed": args.fixed_c}, 0

    y_outer = y[outer_tr]
    g_outer = groups[outer_tr]
    n_splits = min(3, min(Counter(y_outer).values()))
    if n_splits < 2:
        log("WARNING: inner CV impossible; using fixed C=1.0")
        return 1.0, {"fallback_fixed": 1.0}, 0

    inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=4242 + outer_fold)
    scores = {c: [] for c in CANDIDATES}
    conv_total = 0
    for inner_fold, (tr_rel, va_rel) in enumerate(inner.split(np.zeros(len(outer_tr)), y_outer, g_outer)):
        tr = outer_tr[tr_rel]
        va = outer_tr[va_rel]
        t0 = time.time()
        x_tr, x_va, info = fit_transform_features(texts, structs, tr, va)
        log(f"  inner {outer_fold}.{inner_fold}: train={len(tr)} val={len(va)} dim={info['train_shape'][1]} nnz={info['train_nnz']} word_vocab={info['word_vocab']} char_vocab={info['char_vocab']}")
        for c in CANDIDATES:
            clf, conv = fit_lr(x_tr, y[tr], c, 9000 + outer_fold * 10 + inner_fold, args.n_jobs, args.max_iter)
            pred = aligned_proba(clf, x_va, len(va)).argmax(axis=1)
            scores[c].append(macro(y[va], pred))
            conv_total += conv
            del clf
        log(f"  inner {outer_fold}.{inner_fold} done {time.time() - t0:.1f}s")
        del x_tr, x_va
        gc.collect()
    mean_scores = {c: float(np.mean(v)) for c, v in scores.items()}
    selected = max(CANDIDATES, key=lambda c: (mean_scores[c], -c))
    return selected, mean_scores, conv_total


def evaluate(y, probs):
    pred = probs.argmax(axis=1)
    return macro(y, pred), float((pred == y).mean())


def entropy(p):
    pc = np.clip(p, 1e-7, 1.0)
    return -(pc * np.log(pc)).sum(axis=1)


def margin(p):
    part = np.partition(p, -2, axis=1)
    return part[:, -1] - part[:, -2]


def confidence_report(name, mask, pm, ps):
    if not mask.any():
        log(f"{name}: empty")
        return {}
    row = {
        "count": int(mask.sum()),
        "mmbert_entropy": float(entropy(pm[mask]).mean()),
        "mmbert_margin": float(margin(pm[mask]).mean()),
        "mmbert_conf": float(pm[mask].max(axis=1).mean()),
        "sparse_entropy": float(entropy(ps[mask]).mean()),
        "sparse_margin": float(margin(ps[mask]).mean()),
        "sparse_conf": float(ps[mask].max(axis=1).mean()),
    }
    log(f"{name}: {row}")
    return row


def per_class_report(y, pred_m, pred_s):
    _, _, f1_m, support = precision_recall_fscore_support(
        y, pred_m, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    _, _, f1_s, _ = precision_recall_fscore_support(
        y, pred_s, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    rows = []
    for c in range(NUM_CLASSES):
        true_c = y == c
        sparse_only = int(((pred_s == y) & (pred_m != y) & true_c).sum())
        mmbert_only = int(((pred_m == y) & (pred_s != y) & true_c).sum())
        disagree = int((pred_m[true_c] != pred_s[true_c]).sum())
        rows.append({
            "class": CLASSES[c],
            "support": int(support[c]),
            "mmbert_f1": float(f1_m[c]),
            "sparse_f1": float(f1_s[c]),
            "delta_f1": float(f1_s[c] - f1_m[c]),
            "sparse_only_correct": sparse_only,
            "mmbert_only_correct": mmbert_only,
            "disagreement_rate": float(disagree / max(1, int(support[c]))),
        })
    log("per_class_complementarity sorted_by_sparse_only:")
    for row in sorted(rows, key=lambda r: (r["sparse_only_correct"], r["delta_f1"]), reverse=True):
        log(str(row))
    log("candidate_classes_sparse_f1_gt_mmbert:")
    for row in sorted([r for r in rows if r["sparse_f1"] > r["mmbert_f1"]], key=lambda r: r["delta_f1"], reverse=True):
        log(str(row))
    return rows


def blend_tables(y, pm, ps, base_macro):
    prob_rows = []
    log("exploratory same-OOF probability blend scan")
    for alpha in [0.80, 0.85, 0.90, 0.95, 0.975]:
        p = alpha * pm + (1.0 - alpha) * ps
        f1, acc = evaluate(y, p)
        row = {"alpha": alpha, "macro_f1": f1, "accuracy": acc, "delta_vs_mmbert": f1 - base_macro}
        prob_rows.append(row)
        log(str(row))

    log_rows = []
    lm = np.log(np.clip(pm, 1e-7, 1.0))
    ls = np.log(np.clip(ps, 1e-7, 1.0))
    log("exploratory same-OOF log-probability blend scan")
    for beta in [0.05, 0.10, 0.15, 0.20, 0.30]:
        pred = (lm + beta * ls).argmax(axis=1)
        f1 = macro(y, pred)
        acc = float((pred == y).mean())
        row = {"beta": beta, "macro_f1": f1, "accuracy": acc, "delta_vs_mmbert": f1 - base_macro}
        log_rows.append(row)
        log(str(row))
    return prob_rows, log_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/original_data")
    parser.add_argument("--mmbert_oof", default=MMBERT_OOF)
    parser.add_argument("--out", default=SPARSE_OOF)
    parser.add_argument("--selection", choices=["nested", "fixed"], default="nested")
    parser.add_argument("--fixed_c", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--n_jobs", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()

    t_start = time.time()
    log(f"python {platform.python_version()}")
    log(f"cpu_count {os.cpu_count()}")
    try:
        import psutil
        mem = psutil.virtual_memory()
        log(f"ram_available_gb {mem.available / 1e9:.2f} total_gb {mem.total / 1e9:.2f}")
    except Exception as e:
        log(f"ram_available_gb unavailable: {e}")
    log(f"sklearn {sklearn.__version__}")
    log(f"scipy {scipy.__version__}")
    log(f"LogisticRegression config solver=saga penalty=l2 max_iter={args.max_iter} class_weight=None n_jobs={args.n_jobs}")

    samples, ids, texts, structs, y, groups = load_aligned_data(args.data_dir)
    pm = load_mmbert(args.mmbert_oof, len(y))
    lengths = np.array([len(t) for t in texts])
    log("alignment_method train.jsonl row order via src.data.load_train; labels joined by id; OOF filled at SGKF validation indices")
    log("fold_split_method StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)")
    log(f"n_samples {len(y)}")
    log(f"mmBERT_OOF_shape {pm.shape}")
    log(f"label_count {len(y)}")
    log(f"group_count {len(set(groups))}")
    log(f"class_order {CLASSES}")
    log(f"text_count {len(texts)}")
    log(f"text_len_stats min={lengths.min()} p50={np.percentile(lengths, 50):.0f} p90={np.percentile(lengths, 90):.0f} p99={np.percentile(lengths, 99):.0f} max={lengths.max()}")
    log("WARNING: no reproducible current-run OOF bias tuner found; skipping bias application and not modifying submit/model/bias.npy")

    pred_m = pm.argmax(axis=1)
    base_macro, base_acc = evaluate(y, pm)
    if abs(base_macro - 0.770729) > 0.005:
        raise RuntimeError(f"alignment suspected wrong: mmBERT macro-F1 {base_macro:.6f}")
    log(f"mmbert_raw macro_f1={base_macro:.6f} accuracy={base_acc:.6f}")

    oof = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    selected_cs = []
    fold_infos = []
    conv_total = 0
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        t0 = time.time()
        log(f"\n== FOLD {fold} train={len(tr)} val={len(va)} train_groups={len(set(groups[tr]))} val_groups={len(set(groups[va]))} ==")
        log(f"  train_dist {class_dist(y[tr])}")
        log(f"  val_dist {class_dist(y[va])}")
        c, inner_scores, conv_inner = run_inner_selection(texts, structs, y, groups, tr, fold, args)
        conv_total += conv_inner
        selected_cs.append(c)
        log(f"  selected_C={c} inner_scores={inner_scores}")
        x_tr, x_va, info = fit_transform_features(texts, structs, tr, va)
        log(f"  word_vocab={info['word_vocab']} char_vocab={info['char_vocab']} struct_dim={info['struct_dim']}")
        log(f"  train_shape={info['train_shape']} val_shape={info['val_shape']} train_nnz={info['train_nnz']} val_nnz={info['val_nnz']} dtype={info['dtype']}")
        clf, conv = fit_lr(x_tr, y[tr], c, 1000 + fold, args.n_jobs, args.max_iter)
        conv_total += conv
        oof[va] = aligned_proba(clf, x_va, len(va))
        fold_f1 = macro(y[va], oof[va].argmax(axis=1))
        log(f"  fold_macro_f1={fold_f1:.6f} convergence_warnings_outer={conv} elapsed_s={time.time() - t0:.1f}")
        fold_infos.append({"fold": fold, "selected_c": c, "inner_scores": inner_scores, "fold_macro_f1": float(fold_f1), **info})
        del x_tr, x_va, clf
        gc.collect()

    if not np.allclose(oof.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError("sparse OOF probability rows do not sum to 1")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, oof.astype(np.float32))
    log(f"saved {args.out} shape={oof.shape} dtype=float32")

    pred_s = oof.argmax(axis=1)
    sparse_macro, sparse_acc = evaluate(y, oof)
    log("\nSTANDALONE")
    log(f"mmbert_raw macro_f1={base_macro:.6f} accuracy={base_acc:.6f}")
    log(f"sparse_tfidf_lr macro_f1={sparse_macro:.6f} accuracy={sparse_acc:.6f} delta_vs_mmbert={sparse_macro - base_macro:+.6f}")
    pr, rc, f1s, support = precision_recall_fscore_support(y, pred_s, labels=list(range(NUM_CLASSES)), zero_division=0)
    log("sparse_per_class precision recall f1 support")
    for i, cls in enumerate(CLASSES):
        log(f"  {cls:16s} p={pr[i]:.6f} r={rc[i]:.6f} f1={f1s[i]:.6f} support={int(support[i])}")

    disagree = pred_m != pred_s
    both_correct = (pred_m == y) & (pred_s == y)
    mmbert_only = (pred_m == y) & (pred_s != y)
    sparse_only = (pred_m != y) & (pred_s == y)
    both_wrong = (pred_m != y) & (pred_s != y)
    quadrants = {
        "both_correct": int(both_correct.sum()),
        "mmbert_only_correct": int(mmbert_only.sum()),
        "sparse_only_correct": int(sparse_only.sum()),
        "both_wrong": int(both_wrong.sum()),
    }
    log("\nDIVERSITY")
    log(f"disagreement_ratio {disagree.mean():.6f}")
    for k, v in quadrants.items():
        log(f"{k} {v} ratio={v / len(y):.6f}")
    oracle_pred = pred_m.copy()
    oracle_pred[sparse_only] = pred_s[sparse_only]
    oracle_acc = float(((pred_m == y) | (pred_s == y)).mean())
    oracle_macro = macro(y, oracle_pred)
    log(f"top1_oracle_accuracy {oracle_acc:.6f}")
    log(f"oracle_upper_bound_macro_f1 {oracle_macro:.6f} non_deployable=True")

    per_class = per_class_report(y, pred_m, pred_s)
    log("\nCONFIDENCE_DIAGNOSTICS")
    conf = {
        "both_correct": confidence_report("both_correct", both_correct, pm, oof),
        "mmbert_only": confidence_report("mmbert_only", mmbert_only, pm, oof),
        "sparse_only": confidence_report("sparse_only", sparse_only, pm, oof),
        "both_wrong": confidence_report("both_wrong", both_wrong, pm, oof),
    }
    prob_blends, log_blends = blend_tables(y, pm, oof, base_macro)

    best_prob = max(prob_blends, key=lambda r: r["delta_vs_mmbert"])
    best_log = max(log_blends, key=lambda r: r["delta_vs_mmbert"])
    sparse_only_ratio = float(sparse_only.mean())
    oracle_gain = oracle_macro - base_macro
    blend_gain = max(best_prob["delta_vs_mmbert"], best_log["delta_vs_mmbert"])
    if sparse_macro >= 0.74:
        conclusion = "major result; investigate sparse branch as first-class ensemble component"
    elif blend_gain >= 0.003:
        conclusion = "strong complementary branch; prioritize robust cross-fitted blending and submission integration"
    elif blend_gain >= 0.001:
        conclusion = "weak but usable lexical prior"
    elif sparse_only_ratio >= 0.02 and oracle_gain > 0.005:
        conclusion = "branch has diversity; next step should be leakage-safe mmBERT+sparse stacking/gating"
    else:
        conclusion = "sparse branch not useful; stop sparse lexical direction"
    log(f"\nselected_C_per_fold {selected_cs}")
    log(f"convergence_warning_count {conv_total}")
    log(f"conclusion {conclusion}")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_s": time.time() - t_start,
        "selection": args.selection,
        "selected_c_per_fold": selected_cs,
        "convergence_warning_count": conv_total,
        "mmbert": {"macro_f1": base_macro, "accuracy": base_acc},
        "sparse": {"macro_f1": sparse_macro, "accuracy": sparse_acc},
        "disagreement_ratio": float(disagree.mean()),
        "quadrants": {k: {"count": v, "ratio": v / len(y)} for k, v in quadrants.items()},
        "oracle_accuracy": oracle_acc,
        "oracle_macro_f1_upper_bound": oracle_macro,
        "probability_blends": prob_blends,
        "log_probability_blends": log_blends,
        "per_class_complementarity": per_class,
        "confidence": conf,
        "folds": fold_infos,
        "conclusion": conclusion,
    }
    os.makedirs("results", exist_ok=True)
    summary_path = "results/sparse_tfidf_lr_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"saved {summary_path}")
    log(f"finished_at {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
