"""
mmBERT-base 5-fold 파인튜닝.
- StratifiedGroupKFold (session 그룹 + 클래스 층화)
- fold별 best(val macro-F1) 체크포인트 저장 (fp16, save_pretrained)
- OOF 확률 저장 (oof/{tag}.npy, 70000x14) + meta (id/label/fold)
- 최종 OOF macro-F1 리포트
"""
import os, json, argparse, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

from src.data import load_train, CLASSES, NUM_CLASSES

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

@torch.inference_mode()
def predict_proba(model, loader, device):
    model.eval(); out = []
    for batch in loader:
        labels = batch.pop("labels", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits.float()
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--tag", default="mmbert_base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", action="store_true", help="CE에 class_weight 적용")
    ap.add_argument("--num_workers", type=int, default=3)
    args = ap.parse_args()
    print("ARGS:", vars(args))
    set_seed(args.seed)
    device = "cuda"

    ids, texts, ys, groups = load_train(args.data_dir)
    ids, texts, ys, groups = np.array(ids), np.array(texts, dtype=object), np.array(ys), np.array(groups)
    print(f"loaded {len(ids)} rows | {len(set(groups))} sessions | classes={NUM_CLASSES}")

    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs("oof", exist_ok=True)
    ckpt_root = os.path.join("checkpoints", args.tag)
    os.makedirs(ckpt_root, exist_ok=True)

    oof = np.zeros((len(ids), NUM_CLASSES), dtype=np.float32)
    fold_arr = np.full(len(ids), -1, dtype=np.int32)

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr, va) in enumerate(sgkf.split(texts, ys, groups)):
        t0 = time.time()
        print(f"\n===== FOLD {fold} | train {len(tr)} | val {len(va)} =====")
        fold_arr[va] = fold

        # class weight (train fold 기준)
        cw = None
        if args.class_weight:
            classes_present = np.arange(NUM_CLASSES)
            w = compute_class_weight("balanced", classes=np.unique(ys[tr]), y=ys[tr])
            cw = np.ones(NUM_CLASSES, dtype=np.float32)
            for c, wi in zip(np.unique(ys[tr]), w): cw[c] = wi
            cw = torch.tensor(cw, device=device)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=NUM_CLASSES, attn_implementation="eager"
        ).to(device)

        tr_loader = DataLoader(DS(texts[tr].tolist(), ys[tr].tolist(), tok, args.max_len),
                               batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
        va_loader = DataLoader(DS(texts[va].tolist(), ys[va].tolist(), tok, args.max_len),
                               batch_size=args.batch*2, shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(tr_loader) * args.epochs
        sched = get_cosine_schedule_with_warmup(opt, int(total_steps*args.warmup), total_steps)
        lossf = torch.nn.CrossEntropyLoss(weight=cw)

        best_f1, best_state = -1.0, None
        for ep in range(args.epochs):
            model.train()
            for batch in tr_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(**batch).logits
                    loss = lossf(logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()

            proba = predict_proba(model, va_loader, device)
            f1 = f1_score(ys[va], proba.argmax(1), labels=list(range(NUM_CLASSES)),
                          average="macro", zero_division=0)
            print(f"  epoch {ep}: val macroF1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.detach().half().cpu().clone() for k, v in model.state_dict().items()}
                best_proba = proba

        # best 복원 후 fp16 저장 (config+토크나이저 포함)
        model.load_state_dict({k: v.to(device).float() for k, v in best_state.items()})
        fold_dir = os.path.join(ckpt_root, f"fold{fold}")
        model.half().save_pretrained(fold_dir); tok.save_pretrained(fold_dir)
        oof[va] = best_proba
        print(f"  FOLD {fold} best macroF1={best_f1:.4f} | saved {fold_dir} | {time.time()-t0:.0f}s")
        del model; torch.cuda.empty_cache()

    # OOF 저장 + 리포트
    np.save(f"oof/{args.tag}.npy", oof)
    import pandas as pd
    pd.DataFrame({"id": ids, "label": ys, "fold": fold_arr}).to_parquet(f"oof/{args.tag}_meta.parquet")
    oof_pred = oof.argmax(1)
    macro = f1_score(ys, oof_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    acc = (oof_pred == ys).mean()
    print(f"\n===== OOF macroF1={macro:.4f} | acc={acc:.4f} =====")
    print(classification_report(ys, oof_pred, labels=list(range(NUM_CLASSES)),
                                target_names=CLASSES, digits=3, zero_division=0))

if __name__ == "__main__":
    main()
