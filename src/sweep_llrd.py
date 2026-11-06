"""
LLRD (layer-wise lr decay) + EMA (weight averaging) 실험.
- LLRD: 헤드/상위층 큰 lr, 하위층으로 갈수록 decay^depth
- EMA: 매 step 가중치 이동평균 유지, 평가는 EMA 가중치로
비교 기준: fields+640 현 레시피 OOF 0.7735
"""
import os, argparse, random, time, re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from src.data import load_train, NUM_CLASSES

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

def llrd_groups(model, base_lr, decay, weight_decay=0.01):
    """이름에서 layers.N 파싱해 깊이별 lr 차등. 헤드/분류기=base_lr, 아래로 decay^k."""
    # 층 수 파악
    max_layer = -1
    for n, _ in model.named_parameters():
        m = re.search(r"layers\.(\d+)\.", n)
        if m: max_layer = max(max_layer, int(m.group(1)))
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        m = re.search(r"layers\.(\d+)\.", n)
        if m:
            depth_from_top = max_layer - int(m.group(1))
            lr = base_lr * (decay ** (depth_from_top + 1))
        elif "embed" in n:
            lr = base_lr * (decay ** (max_layer + 2))   # 임베딩 = 제일 아래
        else:
            lr = base_lr                                 # 헤드/분류기/최종norm
        key = (lr, 0.0 if ("bias" in n or "norm" in n.lower()) else weight_decay)
        groups.setdefault(key, []).append(p)
    return [{"params": v, "lr": k[0], "weight_decay": k[1]} for k, v in groups.items()]

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def swap_in(self, model):
        self.backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        for n, p in model.named_parameters():
            if n in self.shadow: p.copy_(self.shadow[n])
    @torch.no_grad()
    def swap_out(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: p.copy_(self.backup[n])

@torch.inference_mode()
def proba(model, loader, device):
    model.eval(); out = []
    for b in loader:
        b.pop("labels", None)
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo = model(**b).logits.float()
        out.append(torch.softmax(lo, -1).cpu().numpy())
    return np.concatenate(out, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--llrd", type=float, default=0.0, help="0이면 미사용, 0.9 등이면 층별 decay")
    ap.add_argument("--ema", type=float, default=0.0, help="0이면 미사용, 0.999 등이면 EMA decay")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--tag", default="llrd_ema")
    args = ap.parse_args()
    print("ARGS:", vars(args)); set_seed(args.seed); device = "cuda"

    ids, texts, ys, groups = load_train(args.data_dir)
    ids, texts, ys, groups = np.array(ids), np.array(texts, dtype=object), np.array(ys), np.array(groups)
    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs("oof", exist_ok=True)
    oof_by_ep = {e: np.zeros((len(ids), NUM_CLASSES), np.float32) for e in range(1, args.max_epochs + 1)}

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (tr, va) in enumerate(sgkf.split(texts, ys, groups)):
        t0 = time.time(); print(f"\n== FOLD {fold} | train {len(tr)} val {len(va)} ==")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=NUM_CLASSES, attn_implementation="eager").to(device)
        trl = DataLoader(DS(texts[tr].tolist(), ys[tr].tolist(), tok, args.max_len),
                         batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                         pin_memory=True, drop_last=True)
        val = DataLoader(DS(texts[va].tolist(), ys[va].tolist(), tok, args.max_len),
                         batch_size=args.batch * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        if args.llrd > 0:
            opt = torch.optim.AdamW(llrd_groups(model, args.lr, args.llrd))
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = len(trl) * args.max_epochs
        sch = get_cosine_schedule_with_warmup(opt, int(steps * args.warmup), steps)
        lossf = torch.nn.CrossEntropyLoss()
        ema = EMA(model, args.ema) if args.ema > 0 else None

        for ep in range(1, args.max_epochs + 1):
            model.train()
            for b in trl:
                y = b.pop("labels").to(device); b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = lossf(model(**b).logits.float(), y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
                if ema: ema.update(model)
            if ema: ema.swap_in(model)
            p = proba(model, val, device); oof_by_ep[ep][va] = p
            f1 = f1_score(ys[va], p.argmax(1), labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
            print(f"  ep{ep}: val macroF1={f1:.4f}" + (" (EMA)" if ema else ""))
            if ema: ema.swap_out(model)
        del model; torch.cuda.empty_cache()
        print(f"  fold {fold} done {time.time() - t0:.0f}s")

    print(f"\n===== SWEEP ({args.tag}) OOF macroF1 =====")
    best = (-1, None)
    for ep in range(1, args.max_epochs + 1):
        pred = oof_by_ep[ep].argmax(1)
        f1 = f1_score(ys, pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
        print(f"  epoch {ep}: OOF macroF1={f1:.4f} | acc={(pred == ys).mean():.4f}")
        np.save(f"oof/{args.tag}_ep{ep}.npy", oof_by_ep[ep])
        if f1 > best[0]: best = (f1, ep)
    print(f">>> BEST: epoch {best[1]} -> OOF macroF1={best[0]:.4f}")

if __name__ == "__main__":
    main()
