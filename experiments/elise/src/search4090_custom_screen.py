import argparse
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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from src.data import CLASSES, NUM_CLASSES, load_train
from src.data_flow_variants import load_train_flow_variant


INSPECT_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]
INSPECT_IDS = {CLASSES.index(name): i for i, name in enumerate(INSPECT_CLASSES)}


def log(msg: str = ""):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_value(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return None


def section_intervals(text: str):
    pos = 0
    prompt = []
    flow_state_open = []
    history = []
    in_history = False
    for line in text.splitlines(keepends=True):
        start, end = pos, pos + len(line)
        stripped = line.rstrip("\n")
        if stripped.startswith("[HISTORY]"):
            in_history = True
            history.append((start, end))
        elif in_history:
            history.append((start, end))
        elif stripped.startswith("[PROMPT]"):
            prompt.append((start, end))
        elif stripped.startswith("[FLOW]") or stripped.startswith("[STATE]") or stripped.startswith("[OPEN]"):
            flow_state_open.append((start, end))
        pos = end
    return prompt, flow_state_open, history


def overlap_mask(offsets, intervals):
    mask = []
    for start, end in offsets:
        if start == end:
            mask.append(0)
            continue
        hit = any(start < b and end > a for a, b in intervals)
        mask.append(1 if hit else 0)
    return mask


class SectionDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = texts
        self.labels = labels
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping").squeeze(0).tolist()
        prompt, fso, hist = section_intervals(text)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        label = int(self.labels[idx])
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["aux_labels"] = torch.tensor(INSPECT_IDS.get(label, -100), dtype=torch.long)
        item["prompt_mask"] = torch.tensor(overlap_mask(offsets, prompt), dtype=torch.float32)
        item["fso_mask"] = torch.tensor(overlap_mask(offsets, fso), dtype=torch.float32)
        item["history_mask"] = torch.tensor(overlap_mask(offsets, hist), dtype=torch.float32)
        return item


def masked_mean(hidden, mask):
    mask = mask.to(hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


class CustomModel(nn.Module):
    def __init__(self, model_name, arch, lambda_aux=0.2):
        super().__init__()
        self.arch = arch
        self.lambda_aux = lambda_aux
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, NUM_CLASSES)
        if arch == "prompt_residual":
            self.alpha = nn.Parameter(torch.zeros(()))
        elif arch == "section_static":
            self.fuse = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU())
        elif arch == "inspect_aux":
            self.aux_classifier = nn.Linear(hidden, len(INSPECT_CLASSES))
        elif arch != "global":
            raise ValueError(f"unknown arch: {arch}")

    def forward(self, input_ids, attention_mask, prompt_mask=None, fso_mask=None, history_mask=None, labels=None, aux_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        global_vec = masked_mean(hidden, attention_mask.float())
        if self.arch == "global" or self.arch == "inspect_aux":
            pooled = global_vec
        elif self.arch == "prompt_residual":
            prompt_vec = masked_mean(hidden, prompt_mask.clamp(max=attention_mask.float()))
            pooled = global_vec + torch.sigmoid(self.alpha) * prompt_vec
        elif self.arch == "section_static":
            prompt_vec = masked_mean(hidden, prompt_mask.clamp(max=attention_mask.float()))
            fso_vec = masked_mean(hidden, fso_mask.clamp(max=attention_mask.float()))
            hist_vec = masked_mean(hidden, history_mask.clamp(max=attention_mask.float()))
            pooled = self.fuse(torch.cat([prompt_vec, fso_vec, hist_vec], dim=-1))
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.float(), labels)
            if self.arch == "inspect_aux":
                aux_logits = self.aux_classifier(pooled)
                aux_loss = nn.functional.cross_entropy(aux_logits.float(), aux_labels, ignore_index=-100)
                if torch.isfinite(aux_loss):
                    loss = loss + self.lambda_aux * aux_loss
        return {"loss": loss, "logits": logits}


@torch.inference_mode()
def logits_and_probs(model, loader, device):
    model.eval()
    logits_out = []
    probs_out = []
    for batch in loader:
        batch.pop("labels", None)
        batch.pop("aux_labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch)["logits"].float()
        logits_out.append(logits.cpu().numpy())
        probs_out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(logits_out, axis=0), np.concatenate(probs_out, axis=0)


def fold_ids_or_create(y, groups, split_seed, folds):
    path = f"oof/sgkf{folds}_split_seed{split_seed}_fold_ids.npy"
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    for fold, (_, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold_ids[va] = fold
    if os.path.exists(path):
        old = np.load(path)
        if not np.array_equal(old, fold_ids):
            raise RuntimeError(f"fold assignment mismatch: {path}")
    else:
        os.makedirs("oof", exist_ok=True)
        np.save(path, fold_ids)
    return fold_ids


def macro(y, pred):
    return float(f1_score(y, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))


def status_from_delta(delta):
    if delta < 0.001:
        return "REJECT"
    if delta < 0.002:
        return "FOLD1_CONFIRM"
    if delta >= 0.004:
        return "HIGH_PRIORITY_PROMOTION"
    return "PROMOTE_FULL_CV"


def update_json_list(path, row, key_fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            try:
                rows = json.load(f)
            except json.JSONDecodeError:
                rows = []
    rows = [r for r in rows if any(r.get(k) != row.get(k) for k in key_fields)]
    rows.append(row)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def train_one(args, arch, tag):
    out_dir = os.path.join("oof", "screens", tag)
    summary_path = os.path.join("results", "screens", f"{tag}_summary.json")
    probs_path = os.path.join(out_dir, f"fold{args.fold}_probs.npy")
    logits_path = os.path.join(out_dir, f"fold{args.fold}_logits.npy")
    val_ids_path = os.path.join(out_dir, f"fold{args.fold}_val_ids.npy")
    if os.path.exists(summary_path) and os.path.exists(probs_path) and not args.force:
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)

    base_ids, _, base_y, base_groups = load_train(args.data_dir)
    ids, texts, y, groups, _ = load_train_flow_variant(args.data_dir, args.variant)
    if base_ids != ids or list(base_y) != list(y) or list(base_groups) != list(groups):
        raise RuntimeError("loader changed row, label, or group alignment")
    ids = np.array(ids, dtype=object)
    texts = np.array(texts, dtype=object)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups)
    fold_ids = fold_ids_or_create(y, groups, args.split_seed, args.folds)
    va = np.where(fold_ids == args.fold)[0]
    tr = np.where(fold_ids != args.fold)[0]

    set_seed(args.model_seed)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = CustomModel(args.model, arch, lambda_aux=args.lambda_aux).to(device)
    trl = DataLoader(
        SectionDS(texts[tr].tolist(), y[tr].tolist(), tok, args.max_len),
        batch_size=args.micro_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val = DataLoader(
        SectionDS(texts[va].tolist(), y[va].tolist(), tok, args.max_len),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    update_steps = (len(trl) // args.grad_accum) * args.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(update_steps * args.warmup), update_steps)
    metrics = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        running = 0.0
        stepped = 0
        for i, batch in enumerate(trl, 1):
            labels = batch.pop("labels").to(device)
            aux_labels = batch.pop("aux_labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch, labels=labels, aux_labels=aux_labels)["loss"] / args.grad_accum
            loss.backward()
            running += float(loss.item()) * args.grad_accum
            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
                stepped += 1
        lo, pr = logits_and_probs(model, val, device)
        pred = pr.argmax(axis=1)
        f1 = macro(y[va], pred)
        acc = float((pred == y[va]).mean())
        metrics.append({"epoch": ep, "macro_f1": f1, "accuracy": acc, "loss": running / max(1, len(trl))})
        log(f"{tag} fold{args.fold} ep{ep}: macro_f1={f1:.6f} acc={acc:.6f} updates={stepped}")
        if ep == args.primary_epoch:
            os.makedirs(out_dir, exist_ok=True)
            np.save(logits_path, lo.astype(np.float32))
            np.save(probs_path, pr.astype(np.float32))
            np.save(val_ids_path, ids[va], allow_pickle=True)
    summary = {
        "tag": tag,
        "arch": arch,
        "variant": args.variant,
        "fold": args.fold,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "branch": git_value(["git", "branch", "--show-current"]),
        "model": args.model,
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "primary_epoch": args.primary_epoch,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.micro_batch * args.grad_accum,
        "max_len": args.max_len,
        "lr": args.lr,
        "warmup": args.warmup,
        "lambda_aux": args.lambda_aux,
        "class_order": CLASSES,
        "metrics": metrics,
        "primary_macro_f1": metrics[args.primary_epoch - 1]["macro_f1"],
        "primary_accuracy": metrics[args.primary_epoch - 1]["accuracy"],
        "seconds": time.time() - t0,
        "val_size": int(len(va)),
        "artifact_paths": {"probs": probs_path, "logits": logits_path, "val_ids": val_ids_path},
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--variant", default="flow_exact")
    ap.add_argument("--candidate_arch", required=True, choices=["prompt_residual", "section_static", "inspect_aux"])
    ap.add_argument("--candidate_tag", required=True)
    ap.add_argument("--reference_tag", default="search4090_pool_global_flow_exact")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--model_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--primary_epoch", type=int, default=3)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--val_batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--lambda_aux", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    log("ARGS: " + json.dumps(vars(args), sort_keys=True))
    log(f"torch {torch.__version__} transformers {transformers.__version__}")
    log(f"gpu {torch.cuda.get_device_name(0)}")

    ref = train_one(args, "global", args.reference_tag)
    cand = train_one(args, args.candidate_arch, args.candidate_tag)
    delta = cand["primary_macro_f1"] - ref["primary_macro_f1"]
    status = status_from_delta(delta)
    decision = {
        "candidate": args.candidate_tag,
        "arch": args.candidate_arch,
        "variant": args.variant,
        "status": status,
        "fold": args.fold,
        "fold0_delta": delta if args.fold == 0 else None,
        "paired_reference": args.reference_tag,
        "reference_macro_f1": ref["primary_macro_f1"],
        "candidate_macro_f1": cand["primary_macro_f1"],
        "candidate_accuracy": cand["primary_accuracy"],
        "reason": f"ep{args.primary_epoch} fold{args.fold} delta={delta:+.6f}",
        "recipe": vars(args),
        "artifact_paths": cand["artifact_paths"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    update_json_list("results/search4090_screen_manifest.json", decision, ["candidate", "fold"])
    if status in {"FOLD1_CONFIRM", "PROMOTE_FULL_CV", "HIGH_PRIORITY_PROMOTION", "DIVERSITY_PROMOTION"}:
        update_json_list("results/search4090_promotions.json", decision, ["candidate", "fold"])
    log("DECISION " + json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
