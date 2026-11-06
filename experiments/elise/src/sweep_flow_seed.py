import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import transformers
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from src.data import CLASSES, NUM_CLASSES, load_train
from src.data_flow import flow_stats, load_train_flow


def log(msg: str = ""):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_strings(values) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def git_value(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return None


class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


@torch.inference_mode()
def logits_and_probs(model, loader, device):
    model.eval()
    logits_out = []
    probs_out = []
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.float()
        logits_out.append(logits.cpu().numpy())
        probs_out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(logits_out, axis=0), np.concatenate(probs_out, axis=0)


def compute_or_check_folds(y, groups, split_seed, folds, out_dir):
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    split_rows = []
    for fold, (_, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold_ids[va] = fold
        split_rows.append((fold, int(len(va))))
    if (fold_ids < 0).any():
        raise RuntimeError("fold assignment incomplete")
    os.makedirs(out_dir, exist_ok=True)
    fold_path = os.path.join(out_dir, f"sgkf{folds}_split_seed{split_seed}_fold_ids.npy")
    if os.path.exists(fold_path):
        old = np.load(fold_path)
        if not np.array_equal(old, fold_ids):
            raise RuntimeError(f"fold assignment mismatch: {fold_path}")
    else:
        np.save(fold_path, fold_ids)
        log(f"saved {fold_path}")
    return fold_ids, split_rows, fold_path


def save_train_ids(ids, out_dir):
    path = os.path.join(out_dir, "train_ids.npy")
    arr = np.array(ids, dtype=object)
    if os.path.exists(path):
        old = np.load(path, allow_pickle=True)
        if list(old) != list(arr):
            raise RuntimeError(f"train id order mismatch: {path}")
    else:
        np.save(path, arr, allow_pickle=True)
        log(f"saved {path}")
    return path


def validate_probs(path, arr):
    if arr.shape[1] != NUM_CLASSES:
        raise RuntimeError(f"{path}: bad class dimension {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{path}: non-finite values")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError(f"{path}: rows do not sum to 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--model_seed", type=int, default=45)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.tag is None:
        args.tag = f"fields_flow_seed{args.model_seed}"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.epochs != 4 or args.batch != 64 or args.max_len != 640 or abs(args.lr - 4e-5) > 1e-12:
        raise RuntimeError("recipe guard failed: epochs/batch/max_len/lr changed")

    log("ARGS: " + json.dumps(vars(args), sort_keys=True))
    log(f"torch {torch.__version__} cuda_runtime {torch.version.cuda}")
    log(f"transformers {transformers.__version__}")
    log(f"gpu {torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    device = "cuda"

    base_ids, _, base_y, base_groups = load_train(args.data_dir)
    ids, texts, y, groups, samples = load_train_flow(args.data_dir)
    if base_ids != ids or list(base_y) != list(y) or list(base_groups) != list(groups):
        raise RuntimeError("FLOW loader changed row, label, or group alignment")
    ids = np.array(ids, dtype=object)
    texts = np.array(texts, dtype=object)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups)
    if len(ids) != 70000 or NUM_CLASSES != 14:
        raise RuntimeError(f"unexpected data shape: n={len(ids)} classes={NUM_CLASSES}")

    log("serializer=src.data.serialize_example + inserted [FLOW] line")
    log(f"flow_stats {json.dumps(flow_stats(samples), sort_keys=True, ensure_ascii=False)}")
    train_ids_path = save_train_ids(ids, "oof")
    fold_ids, split_rows, fold_path = compute_or_check_folds(y, groups, args.split_seed, args.folds, "oof")
    train_ids_hash = sha256_strings(ids)
    fold_hash = sha256_array(fold_ids)
    log(f"train_ids_sha256 {train_ids_hash}")
    log(f"fold_assignment_sha256 {fold_hash}")
    log(f"split_rows {split_rows}")

    set_seed(args.model_seed)
    log(f"seed_semantics global_set_seed_once_before_cv={args.model_seed}; no per-fold reseed")
    tok = AutoTokenizer.from_pretrained(args.model)
    logits_by_ep = {ep: np.zeros((len(ids), NUM_CLASSES), dtype=np.float32) for ep in range(1, args.epochs + 1)}
    probs_by_ep = {ep: np.zeros((len(ids), NUM_CLASSES), dtype=np.float32) for ep in range(1, args.epochs + 1)}
    fold_metrics = []

    for fold in range(args.folds):
        va = np.where(fold_ids == fold)[0]
        tr = np.where(fold_ids != fold)[0]
        t0 = time.time()
        log(f"\n== FOLD {fold} train={len(tr)} val={len(va)} model_seed={args.model_seed} ==")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=NUM_CLASSES,
            attn_implementation="eager",
        ).to(device)
        trl = DataLoader(
            DS(texts[tr].tolist(), y[tr].tolist(), tok, args.max_len),
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val = DataLoader(
            DS(texts[va].tolist(), y[va].tolist(), tok, args.max_len),
            batch_size=args.batch * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = len(trl) * args.epochs
        sch = get_cosine_schedule_with_warmup(opt, int(steps * args.warmup), steps)
        loss_fn = torch.nn.CrossEntropyLoss()
        for ep in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for batch in trl:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(**batch).logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                running += float(loss.item())
            lo, pr = logits_and_probs(model, val, device)
            logits_by_ep[ep][va] = lo.astype(np.float32)
            probs_by_ep[ep][va] = pr.astype(np.float32)
            pred = pr.argmax(axis=1)
            f1 = f1_score(y[va], pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
            acc = float((pred == y[va]).mean())
            fold_metrics.append({
                "fold": fold,
                "epoch": ep,
                "macro_f1": float(f1),
                "accuracy": acc,
                "loss": running / max(1, len(trl)),
                "seconds": time.time() - t0,
            })
            log(f"  epoch {ep}: loss={running / max(1, len(trl)):.6f} macro_f1={f1:.6f} accuracy={acc:.6f}")
        fold_dir = os.path.join("checkpoints", args.tag, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        model.half().save_pretrained(fold_dir)
        if fold == 0:
            tok.save_pretrained(os.path.join("checkpoints", args.tag, "tokenizer"))
        log(f"  saved checkpoint {fold_dir}")
        log(f"  fold {fold} done seconds={time.time() - t0:.1f}")
        del model
        torch.cuda.empty_cache()

    epoch_scores = []
    for ep in range(1, args.epochs + 1):
        logits_path = os.path.join("oof", f"{args.tag}_ep{ep}_logits.npy")
        probs_path = os.path.join("oof", f"{args.tag}_ep{ep}_probs.npy")
        validate_probs(probs_path, probs_by_ep[ep])
        if not np.isfinite(logits_by_ep[ep]).all():
            raise RuntimeError(f"{logits_path}: non-finite logits")
        pred = probs_by_ep[ep].argmax(axis=1)
        f1 = f1_score(y, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
        acc = float((pred == y).mean())
        np.save(logits_path, logits_by_ep[ep])
        np.save(probs_path, probs_by_ep[ep])
        epoch_scores.append({
            "epoch": ep,
            "macro_f1": float(f1),
            "accuracy": acc,
            "logits_path": logits_path,
            "probs_path": probs_path,
        })
        log(f"epoch {ep}: macro_f1={f1:.6f} accuracy={acc:.6f}")
        log(f"saved {logits_path}")
        log(f"saved {probs_path}")

    meta_path = os.path.join("oof", f"{args.tag}_meta.json")
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "branch": git_value(["git", "branch", "--show-current"]),
        "model": args.model,
        "serializer": "src.data_flow.serialize_example_flow",
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "seed_semantics": "global set_seed(model_seed) once before CV; no per-fold reseed, matching FLOW42 sweep_flow.py behavior",
        "folds": args.folds,
        "epochs": args.epochs,
        "batch": args.batch,
        "max_len": args.max_len,
        "lr": args.lr,
        "warmup": args.warmup,
        "class_weight": False,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "precision": "bf16 autocast",
        "attention": "eager",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "train_ids_path": train_ids_path,
        "fold_assignment_path": fold_path,
        "train_ids_sha256": train_ids_hash,
        "fold_assignment_sha256": fold_hash,
        "class_order": CLASSES,
        "fold_metrics": fold_metrics,
        "epoch_scores": epoch_scores,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"saved {meta_path}")


if __name__ == "__main__":
    main()
