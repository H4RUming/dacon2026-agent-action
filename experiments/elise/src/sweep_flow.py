"""
FLOW prefix sweep: same training semantics as src.sweep_epochs, but inputs use
src.data_flow.serialize_example_flow. Supports fixed-fold screening.
"""
import argparse
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import transformers
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from src.data import CLASSES, NUM_CLASSES, load_train, read_jsonl
from src.data_flow import flow_stats, lastres_category, load_train_flow, turn_bucket_flow


BASELINE_FULL = {
    1: {"macro_f1": 0.706828},
    2: {"macro_f1": 0.745692},
    3: {"macro_f1": 0.770729},
    4: {"macro_f1": 0.763974},
}
BASELINE_FOLD0 = {1: 0.7050, 2: 0.7603, 3: 0.7691, 4: 0.7634}
FOCUS_PAIRS = [
    ("read_file", "grep_search"),
    ("read_file", "glob_pattern"),
    ("read_file", "list_directory"),
    ("grep_search", "glob_pattern"),
    ("run_bash", "run_tests"),
    ("run_bash", "lint_or_typecheck"),
    ("run_tests", "lint_or_typecheck"),
    ("edit_file", "apply_patch"),
    ("ask_user", "plan_task"),
    ("plan_task", "web_search"),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


@torch.inference_mode()
def proba(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.float()
        out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(out, 0)


def parse_folds_to_run(raw):
    if raw is None or raw.strip() == "":
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip() != ""}


def token_length_analysis(tok, base_texts, flow_texts, max_len):
    base_lengths = np.array([len(tok(t, truncation=False)["input_ids"]) for t in base_texts])
    flow_lengths = np.array([len(tok(t, truncation=False)["input_ids"]) for t in flow_texts])
    def pack(name, arr):
        return {
            f"{name}_p50": int(np.percentile(arr, 50)),
            f"{name}_p90": int(np.percentile(arr, 90)),
            f"{name}_p95": int(np.percentile(arr, 95)),
            f"{name}_p99": int(np.percentile(arr, 99)),
            f"{name}_max": int(arr.max()),
        }
    stats = {}
    stats.update(pack("baseline", base_lengths))
    stats.update(pack("flow", flow_lengths))
    stats["baseline_truncated_count"] = int((base_lengths > max_len).sum())
    stats["flow_truncated_count"] = int((flow_lengths > max_len).sum())
    stats["newly_truncated_due_to_flow"] = int(((base_lengths <= max_len) & (flow_lengths > max_len)).sum())
    return stats


def macro(y, pred):
    return f1_score(y, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)


def validate_oof(path, arr):
    if arr.shape[1] != NUM_CLASSES:
        raise RuntimeError(f"{path}: bad class dimension {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{path}: non-finite values")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError(f"{path}: rows do not sum to 1")


def print_full_analysis(samples, y, base_probs, flow_probs):
    base_pred = base_probs.argmax(1)
    flow_pred = flow_probs.argmax(1)
    disagree = base_pred != flow_pred
    both_correct = (base_pred == y) & (flow_pred == y)
    base_only = (base_pred == y) & (flow_pred != y)
    flow_only = (base_pred != y) & (flow_pred == y)
    both_wrong = (base_pred != y) & (flow_pred != y)
    print("\nPAIRED_SAMPLE_ANALYSIS", flush=True)
    print(f"prediction_disagreement_ratio {disagree.mean():.6f}", flush=True)
    print(f"both_correct {both_correct.sum()} ratio={both_correct.mean():.6f}", flush=True)
    print(f"baseline_only_correct {base_only.sum()} ratio={base_only.mean():.6f}", flush=True)
    print(f"FLOW_only_correct {flow_only.sum()} ratio={flow_only.mean():.6f}", flush=True)
    print(f"both_wrong {both_wrong.sum()} ratio={both_wrong.mean():.6f}", flush=True)
    print(f"oracle_either_correct_accuracy {((base_pred == y) | (flow_pred == y)).mean():.6f}", flush=True)

    base_f1 = precision_recall_fscore_support(y, base_pred, labels=list(range(NUM_CLASSES)), zero_division=0)[2]
    flow_f1 = precision_recall_fscore_support(y, flow_pred, labels=list(range(NUM_CLASSES)), zero_division=0)[2]
    support = np.bincount(y, minlength=NUM_CLASSES)
    print("\nPER_CLASS_F1", flush=True)
    for i, cls in enumerate(CLASSES):
        print(
            f"{cls:16s} support={support[i]:5d} baseline_f1={base_f1[i]:.6f} "
            f"FLOW_f1={flow_f1[i]:.6f} delta={flow_f1[i] - base_f1[i]:+.6f}",
            flush=True,
        )

    cm_base = confusion_matrix(y, base_pred, labels=list(range(NUM_CLASSES)))
    cm_flow = confusion_matrix(y, flow_pred, labels=list(range(NUM_CLASSES)))
    print("\nFOCUS_CONFUSION_CHANGES true->pred baseline flow delta", flush=True)
    class_to_id = {c: i for i, c in enumerate(CLASSES)}
    for a, b in FOCUS_PAIRS:
        ia, ib = class_to_id[a], class_to_id[b]
        for true_i, pred_i, label in [(ia, ib, f"{a}->{b}"), (ib, ia, f"{b}->{a}")]:
            base = int(cm_base[true_i, pred_i])
            flow = int(cm_flow[true_i, pred_i])
            print(f"{label:32s} {base:5d} {flow:5d} {flow - base:+5d}", flush=True)

    print("\nFLOW_SLICE_ANALYSIS", flush=True)
    actions = [
        len([h for h in (sample.get("history", []) or []) if h.get("role") == "assistant_action"])
        for sample in samples
    ]
    prev_action_slice = np.array(["3+" if n >= 3 else str(n) for n in actions], dtype=object)
    turn_slice = np.array([turn_bucket_flow((s.get("session_meta", {}) or {}).get("turn_index")) for s in samples], dtype=object)
    lastres_slice = np.array([lastres_category(s) for s in samples], dtype=object)
    disagree_slice = np.where(disagree, "disagree", "agree")
    for name, values, order in [
        ("prev_action_count", prev_action_slice, ["0", "1", "2", "3+"]),
        ("turn_bucket", turn_slice, ["0", "1", "2", "3", "4-5", "6-7", "8-11", "12+", "unknown"]),
        ("lastres", lastres_slice, ["none", "fail", "notfound", "pass", "changed", "found", "other"]),
        ("baseline_flow_agreement", disagree_slice, ["agree", "disagree"]),
    ]:
        print(f"\nSLICE {name}", flush=True)
        for val in order:
            mask = values == val
            if not mask.any():
                continue
            bacc = float((base_pred[mask] == y[mask]).mean())
            facc = float((flow_pred[mask] == y[mask]).mean())
            print(f"{val:12s} N={mask.sum():5d} baseline_acc={bacc:.6f} FLOW_acc={facc:.6f} delta={facc - bacc:+.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--folds_to_run", default=None)
    ap.add_argument("--max_epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold_seed", type=int, default=42)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--tag", default="fields_flow")
    args = ap.parse_args()
    print("ARGS:", vars(args), flush=True)
    set_seed(args.seed)
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep")
    print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    print(f"torch {torch.__version__}", flush=True)
    print(f"transformers {transformers.__version__}", flush=True)
    print(f"cuda_available {torch.cuda.is_available()}", flush=True)

    base_ids, base_texts, base_y, base_groups = load_train(args.data_dir)
    ids, texts, ys, groups, samples = load_train_flow(args.data_dir)
    if base_ids != ids or list(base_y) != list(ys) or list(base_groups) != list(groups):
        raise RuntimeError("FLOW loader changed row, label, or group alignment")
    ids = np.array(ids)
    texts = np.array(texts, dtype=object)
    ys = np.array(ys, dtype=np.int64)
    groups = np.array(groups)
    print("inspection_summary active_serializer=src.data.serialize_example + inserted [FLOW] line", flush=True)
    print(f"inspection_summary active_model={args.model} max_len={args.max_len} batch={args.batch} lr={args.lr} warmup={args.warmup} class_weight={args.class_weight}", flush=True)
    print("inspection_summary group_extraction=src.data.session_id(sample_id)", flush=True)
    print(f"inspection_summary fold_splitter=StratifiedGroupKFold(n_splits={args.folds}, shuffle=True, random_state={args.fold_seed})", flush=True)
    print(f"inspection_summary class_order={CLASSES}", flush=True)
    print("inspection_summary OOF row alignment=train.jsonl/load_train row order, validation indices filled in-place", flush=True)
    print(f"loaded {len(ids)} rows groups={len(set(groups))} classes={NUM_CLASSES}", flush=True)
    print("FLOW_STATS", flow_stats(samples), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    token_stats = token_length_analysis(tok, base_texts, texts.tolist(), args.max_len)
    print("TOKEN_LENGTH_ANALYSIS", token_stats, flush=True)

    folds_to_run = parse_folds_to_run(args.folds_to_run)
    partial = folds_to_run is not None
    if partial:
        print(f"PARTIAL_FOLD_RUN folds_to_run={sorted(folds_to_run)}; not a full OOF score", flush=True)
    os.makedirs("oof", exist_ok=True)
    if partial:
        os.makedirs("oof/screens", exist_ok=True)
    oof_by_ep = {e: np.zeros((len(ids), NUM_CLASSES), np.float32) for e in range(1, args.max_epochs + 1)}
    filled = np.zeros(len(ids), dtype=bool)

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.fold_seed)
    fold_metrics = {}
    for fold, (tr, va) in enumerate(sgkf.split(texts, ys, groups)):
        if partial and fold not in folds_to_run:
            continue
        t0 = time.time()
        print(f"\n== FOLD {fold} | train {len(tr)} val {len(va)} ==", flush=True)
        cw = None
        if args.class_weight:
            w = compute_class_weight("balanced", classes=np.unique(ys[tr]), y=ys[tr])
            cw_arr = np.ones(NUM_CLASSES, np.float32)
            for c, wi in zip(np.unique(ys[tr]), w):
                cw_arr[c] = wi
            cw = torch.tensor(cw_arr, device=device)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=NUM_CLASSES, attn_implementation="eager"
        ).to(device)
        trl = DataLoader(
            DS(texts[tr].tolist(), ys[tr].tolist(), tok, args.max_len),
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val = DataLoader(
            DS(texts[va].tolist(), ys[va].tolist(), tok, args.max_len),
            batch_size=args.batch * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = len(trl) * args.max_epochs
        sch = get_cosine_schedule_with_warmup(opt, int(steps * args.warmup), steps)
        lossf = torch.nn.CrossEntropyLoss(weight=cw)
        fold_metrics[fold] = {}
        for ep in range(1, args.max_epochs + 1):
            model.train()
            for batch in trl:
                y = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = lossf(model(**batch).logits.float(), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
            p = proba(model, val, device)
            oof_by_ep[ep][va] = p
            filled[va] = True
            f1 = macro(ys[va], p.argmax(1))
            acc = float((p.argmax(1) == ys[va]).mean())
            fold_metrics[fold][ep] = {"macro_f1": f1, "accuracy": acc}
            ref = BASELINE_FOLD0.get(ep) if fold == 0 else None
            delta = f" delta_vs_ref={f1 - ref:+.4f}" if ref is not None else ""
            print(f"  ep{ep}: val macroF1={f1:.4f} acc={acc:.4f}{delta}", flush=True)
        del model
        torch.cuda.empty_cache()
        print(f"  fold {fold} done {time.time() - t0:.0f}s", flush=True)

    if partial:
        for ep, arr in oof_by_ep.items():
            path = f"oof/screens/{args.tag}_folds{''.join(map(str, sorted(folds_to_run)))}_ep{ep}.npy"
            np.save(path, arr.astype(np.float32))
            print(f"saved partial {path} filled_rows={int(filled.sum())}", flush=True)
        if 0 in fold_metrics and 3 in fold_metrics[0]:
            delta = fold_metrics[0][3]["macro_f1"] - BASELINE_FOLD0[3]
            print(f"STAGE_A_DECISION fold0_ep3_delta={delta:+.6f}", flush=True)
            if delta < 0.001:
                print("screening_decision STOP: weak/failed screen; do not run full 5-fold", flush=True)
            elif delta < 0.002:
                print("screening_decision AMBIGUOUS: run Fold 1 confirmation only", flush=True)
            else:
                print("screening_decision PASS: proceed to full 5-fold", flush=True)
        return

    print(f"\n===== FLOW SWEEP ({args.tag}) FULL OOF =====", flush=True)
    results = []
    best = (-1.0, None)
    for ep in range(1, args.max_epochs + 1):
        arr = oof_by_ep[ep]
        validate_oof(f"oof/{args.tag}_ep{ep}.npy", arr)
        pred = arr.argmax(1)
        f1 = macro(ys, pred)
        acc = float((pred == ys).mean())
        base = BASELINE_FULL.get(ep, {})
        delta = f1 - base["macro_f1"] if "macro_f1" in base else None
        print(f"epoch {ep}: macro_f1={f1:.6f} accuracy={acc:.6f} delta_vs_baseline_macro={delta:+.6f}", flush=True)
        np.save(f"oof/{args.tag}_ep{ep}.npy", arr.astype(np.float32))
        results.append({"epoch": ep, "macro_f1": f1, "accuracy": acc, "delta_vs_baseline_macro": delta})
        if f1 > best[0]:
            best = (f1, ep)
    ep3_delta = results[2]["delta_vs_baseline_macro"] if len(results) >= 3 else None
    print(f"FLOW_PRIMARY_DELTA ep3_delta_macro_f1={ep3_delta:+.6f}", flush=True)
    print(f"FLOW_BEST_EPOCH epoch={best[1]} macro_f1={best[0]:.6f}", flush=True)
    if ep3_delta <= 0.001:
        print("decision no practical gain; stop FLOW direction", flush=True)
    elif ep3_delta < 0.002:
        print("decision weak signal; inspect class/slice effects only", flush=True)
    elif ep3_delta < 0.004:
        print("decision useful representation gain; next candidate FLOW + family auxiliary loss", flush=True)
    elif ep3_delta < 0.006:
        print("decision strong gain; prioritize reproducibility and submission integration later", flush=True)
    else:
        print("decision major lever; stop unrelated experiments and validate immediately", flush=True)

    base_path = "oof/fields_mmbert_ep3.npy"
    if os.path.exists(base_path):
        base_probs = np.load(base_path)
        print_full_analysis(samples, ys, base_probs, oof_by_ep[3])
    else:
        print(f"WARNING missing {base_path}; skipping paired analysis", flush=True)


if __name__ == "__main__":
    main()
