"""학습 정규화/노이즈 loss 통합 스크린. 한 번에 한 기법만 켜서 5-fold OOF.
기준: fields+640 기본 레시피 OOF 0.7735"""
import os, argparse, random, time
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from src.data import load_train, NUM_CLASSES

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len, idxs=None):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len
        self.idxs = idxs if idxs is not None else list(range(len(texts)))
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        item["gidx"] = torch.tensor(self.idxs[i], dtype=torch.long)
        return item

# ---------- losses ----------
def gce_loss(logits, y, q=0.7):
    p = torch.softmax(logits, -1).gather(1, y[:, None]).squeeze(1).clamp(1e-7, 1)
    return ((1 - p.pow(q)) / q).mean()

def sce_loss(logits, y, a=1.0, b=1.0):
    ce = F.cross_entropy(logits, y)
    pred = torch.softmax(logits, -1).clamp(1e-7, 1)
    onehot = F.one_hot(y, logits.size(-1)).float().clamp(1e-4, 1)
    rce = (-pred * onehot.log()).sum(1).mean()
    return a * ce + b * rce

class ELR:
    def __init__(self, n, ncls, lam=3.0, beta=0.7):
        self.t = torch.zeros(n, ncls).cuda(); self.lam, self.beta = lam, beta
    def __call__(self, logits, y, gidx):
        p = torch.softmax(logits, -1).detach()
        self.t[gidx] = self.beta * self.t[gidx] + (1 - self.beta) * p
        ce = F.cross_entropy(logits, y)
        pt = (torch.softmax(logits, -1) * self.t[gidx]).sum(1).clamp(1e-4, 1 - 1e-4)
        return ce + self.lam * torch.log(1 - pt).mul(-1).mean()

# ---------- SAM ----------
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_cls, rho=0.05, **kw):
        super().__init__(params, dict(rho=rho, **kw))
        self.base = base_cls(self.param_groups, **kw)
        self.param_groups = self.base.param_groups
    @torch.no_grad()
    def first_step(self):
        gn = torch.norm(torch.stack([p.grad.norm(2) for g in self.param_groups
                                     for p in g["params"] if p.grad is not None]), 2)
        for g in self.param_groups:
            s = g["rho"] / (gn + 1e-12)
            for p in g["params"]:
                if p.grad is None: continue
                self.state[p]["e_w"] = p.grad * s
                p.add_(self.state[p]["e_w"])
        self.zero_grad(set_to_none=True)
    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"]); self.state[p].pop("e_w")
        self.base.step(); self.zero_grad(set_to_none=True)

# ---------- FGM ----------
class FGM:
    def __init__(self, model, eps=1.0, name="embeddings.tok_embeddings"):
        self.model, self.eps, self.name, self.backup = model, eps, name, {}
    def attack(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and self.name in n and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = p.grad.norm()
                if norm and not torch.isnan(norm):
                    p.data.add_(self.eps * p.grad / norm)
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup: p.data = self.backup[n]
        self.backup = {}

@torch.inference_mode()
def proba(model, loader, device):
    model.eval(); out = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items() if k not in ("labels", "gidx")}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo = model(**b).logits.float()
        out.append(torch.softmax(lo, -1).cpu().numpy())
    return np.concatenate(out, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--max_epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    # 기법 플래그 (한 번에 하나만)
    ap.add_argument("--rdrop", type=float, default=0)   # KL 가중 (1.0 권장)
    ap.add_argument("--sam", type=float, default=0)     # rho (0.05)
    ap.add_argument("--fgm", type=float, default=0)     # eps (1.0)
    ap.add_argument("--gce", type=float, default=0)     # q (0.7)
    ap.add_argument("--sce", type=int, default=0)
    ap.add_argument("--elr", type=int, default=0)
    ap.add_argument("--ls", type=float, default=0)      # label smoothing (0.1)
    ap.add_argument("--swa", type=int, default=0)       # ep2부터 가중치 평균
    ap.add_argument("--hist_drop", type=float, default=0)  # (미구현: 직렬화 수정 필요, 자리만)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(); print("ARGS:", vars(args))
    set_seed(args.seed); device = "cuda"

    ids, texts, ys, groups = load_train("data/original_data")
    texts = np.array(texts, dtype=object); ys = np.array(ys); groups = np.array(groups)
    L = list(range(NUM_CLASSES))
    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs("oof", exist_ok=True)
    oof = {e: np.zeros((len(ids), NUM_CLASSES), np.float32) for e in range(1, args.max_epochs + 1)}

    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(sgkf.split(texts, ys, groups)):
        t0 = time.time(); print(f"\n== FOLD {fold} ==")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=NUM_CLASSES, attn_implementation="eager").to(device)
        trl = DataLoader(DS(texts[tr].tolist(), ys[tr].tolist(), tok, args.max_len, list(tr)),
                         batch_size=args.batch, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
        val = DataLoader(DS(texts[va].tolist(), ys[va].tolist(), tok, args.max_len),
                         batch_size=args.batch * 2, shuffle=False, num_workers=3, pin_memory=True)
        if args.sam > 0:
            opt = SAM(model.parameters(), torch.optim.AdamW, rho=args.sam, lr=args.lr, weight_decay=0.01)
            sch = get_cosine_schedule_with_warmup(opt.base, int(len(trl) * args.max_epochs * args.warmup),
                                                  len(trl) * args.max_epochs)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            sch = get_cosine_schedule_with_warmup(opt, int(len(trl) * args.max_epochs * args.warmup),
                                                  len(trl) * args.max_epochs)
        fgm = FGM(model, args.fgm) if args.fgm > 0 else None
        elr = ELR(len(ids), NUM_CLASSES) if args.elr else None
        swa_sd, swa_n = None, 0

        def calc_loss(logits, y, gidx):
            if args.gce > 0: return gce_loss(logits, y, args.gce)
            if args.sce: return sce_loss(logits, y)
            if args.elr: return elr(logits, y, gidx)
            if args.ls > 0: return F.cross_entropy(logits, y, label_smoothing=args.ls)
            return F.cross_entropy(logits, y)

        for ep in range(1, args.max_epochs + 1):
            model.train()
            for b in trl:
                y = b.pop("labels").to(device); gidx = b.pop("gidx").to(device)
                b = {k: v.to(device) for k, v in b.items()}
                if args.sam > 0:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = calc_loss(model(**b).logits.float(), y, gidx)
                    loss.backward(); opt.first_step()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss2 = calc_loss(model(**b).logits.float(), y, gidx)
                    loss2.backward(); opt.second_step(); sch.step()
                elif args.rdrop > 0:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        l1 = model(**b).logits.float(); l2 = model(**b).logits.float()
                    ce = 0.5 * (F.cross_entropy(l1, y) + F.cross_entropy(l2, y))
                    p, q = F.log_softmax(l1, -1), F.log_softmax(l2, -1)
                    kl = 0.5 * (F.kl_div(p, q.exp(), reduction="batchmean")
                                + F.kl_div(q, p.exp(), reduction="batchmean"))
                    loss = ce + args.rdrop * kl
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sch.step(); opt.zero_grad()
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = calc_loss(model(**b).logits.float(), y, gidx)
                    loss.backward()
                    if fgm:
                        fgm.attack()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss_adv = calc_loss(model(**b).logits.float(), y, gidx)
                        loss_adv.backward(); fgm.restore()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sch.step(); opt.zero_grad()
            if args.swa and ep >= 2:
                sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
                if swa_sd is None: swa_sd = sd; swa_n = 1
                else:
                    for k in swa_sd: swa_sd[k] = (swa_sd[k] * swa_n + sd[k]) / (swa_n + 1)
                    swa_n += 1
            p = proba(model, val, device); oof[ep][va] = p
            f1 = f1_score(ys[va], p.argmax(1), labels=L, average="macro", zero_division=0)
            print(f"  ep{ep}: val macroF1={f1:.4f}")
        if args.swa and swa_sd is not None:
            model.load_state_dict({k: v.to(device) for k, v in swa_sd.items()})
            p = proba(model, val, device)
            f1 = f1_score(ys[va], p.argmax(1), labels=L, average="macro", zero_division=0)
            print(f"  SWA(ep2-{args.max_epochs}): val macroF1={f1:.4f}")
            oof[args.max_epochs][va] = p  # SWA 결과를 마지막 슬롯에 (태그로 구분)
        del model; torch.cuda.empty_cache()
        print(f"  fold {fold} done {time.time()-t0:.0f}s")

    print(f"\n===== {args.tag} OOF =====")
    best = (-1, None)
    for ep in range(1, args.max_epochs + 1):
        f1 = f1_score(ys, oof[ep].argmax(1), labels=L, average="macro", zero_division=0)
        print(f"  epoch {ep}: OOF macroF1={f1:.4f}")
        np.save(f"oof/{args.tag}_ep{ep}.npy", oof[ep])
        if f1 > best[0]: best = (f1, ep)
    print(f">>> BEST: epoch {best[1]} -> {best[0]:.4f}  (기준 0.7735)")

if __name__ == "__main__":
    main()
