"""
CE + SupCon 스윕. mmBERT mean-pool 임베딩 -> projection -> SupCon.
- class_weight OFF 고정
- 매 epoch OOF macroF1 기록 (epoch별 비교)
- lambda(supcon 가중치) 인자로 스윕
"""
import os, argparse, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
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

class Model(nn.Module):
    """mmBERT backbone + mean-pool + (classifier head, projection head)."""
    def __init__(self, name, n_cls, proj_dim=128):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, attn_implementation="eager")
        h = self.enc.config.hidden_size
        self.classifier = nn.Linear(h, n_cls)
        self.proj = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, proj_dim))
    def meanpool(self, last, mask):
        m = mask.unsqueeze(-1).float()
        return (last * m).sum(1) / m.sum(1).clamp(min=1e-6)
    def forward(self, input_ids, attention_mask, **kw):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.meanpool(out.last_hidden_state, attention_mask)   # (B,H)
        logits = self.classifier(emb)
        z = F.normalize(self.proj(emb), dim=-1)                       # (B,proj)
        return logits, z

def supcon_loss(z, labels, temp=0.07):
    """Supervised Contrastive Loss (Khosla et al.). z: (B,d) L2-normed."""
    device = z.device; B = z.size(0)
    sim = z @ z.t() / temp                                   # (B,B)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach() # 수치안정
    logits_mask = ~torch.eye(B, dtype=torch.bool, device=device)
    exp = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    lab = labels.view(-1,1)
    pos_mask = (lab == lab.t()) & logits_mask                # 같은 클래스(자기제외)
    pos_cnt = pos_mask.sum(1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(1)[valid] / pos_cnt[valid]
    return -mean_log_prob_pos.mean()

@torch.inference_mode()
def proba(model, loader, device):
    model.eval(); out=[]
    for b in loader:
        b.pop("labels", None)
        b={k:v.to(device) for k,v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits,_ = model(**b)
        out.append(torch.softmax(logits.float(),-1).cpu().numpy())
    return np.concatenate(out,0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--lam", type=float, default=0.3, help="SupCon 가중치")
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--tag", default="supcon")
    args = ap.parse_args()
    print("ARGS:", vars(args)); set_seed(args.seed); device="cuda"

    ids, texts, ys, groups = load_train(args.data_dir)
    ids, texts, ys, groups = np.array(ids), np.array(texts,dtype=object), np.array(ys), np.array(groups)
    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs("oof", exist_ok=True)
    oof_by_ep = {e: np.zeros((len(ids), NUM_CLASSES), np.float32) for e in range(1, args.max_epochs+1)}

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold,(tr,va) in enumerate(sgkf.split(texts, ys, groups)):
        t0=time.time(); print(f"\n== FOLD {fold} | train {len(tr)} val {len(va)} ==")
        model=Model(args.model, NUM_CLASSES).to(device)
        trl=DataLoader(DS(texts[tr].tolist(),ys[tr].tolist(),tok,args.max_len),
                       batch_size=args.batch,shuffle=True,num_workers=args.num_workers,
                       pin_memory=True,drop_last=True)
        val=DataLoader(DS(texts[va].tolist(),ys[va].tolist(),tok,args.max_len),
                       batch_size=args.batch*2,shuffle=False,num_workers=args.num_workers,pin_memory=True)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
        steps=len(trl)*args.max_epochs
        sch=get_cosine_schedule_with_warmup(opt,int(steps*args.warmup),steps)
        ce=nn.CrossEntropyLoss()
        for ep in range(1, args.max_epochs+1):
            model.train()
            for b in trl:
                y=b.pop("labels").to(device); b={k:v.to(device) for k,v in b.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, z = model(**b)
                    loss = ce(logits.float(), y) + args.lam * supcon_loss(z.float(), y, args.temp)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sch.step(); opt.zero_grad()
            p=proba(model,val,device); oof_by_ep[ep][va]=p
            f1=f1_score(ys[va],p.argmax(1),labels=list(range(NUM_CLASSES)),average="macro",zero_division=0)
            print(f"  ep{ep}: val macroF1={f1:.4f}")
        del model; torch.cuda.empty_cache()
        print(f"  fold {fold} done {time.time()-t0:.0f}s")

    print(f"\n===== SUPCON SWEEP (lam={args.lam}) OOF macroF1 =====")
    best=(-1,None)
    for ep in range(1, args.max_epochs+1):
        pred=oof_by_ep[ep].argmax(1)
        f1=f1_score(ys,pred,labels=list(range(NUM_CLASSES)),average="macro",zero_division=0)
        acc=(pred==ys).mean()
        print(f"  epoch {ep}: OOF macroF1={f1:.4f} | acc={acc:.4f}")
        np.save(f"oof/{args.tag}_lam{args.lam}_ep{ep}.npy", oof_by_ep[ep])
        if f1>best[0]: best=(f1,ep)
    print(f">>> BEST(lam={args.lam}): epoch {best[1]} -> OOF macroF1={best[0]:.4f}")

if __name__=="__main__":
    main()
