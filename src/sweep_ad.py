"""
epoch 스윕: 한 번 학습하며 매 epoch OOF를 기록 -> epoch별 macroF1 비교.
5-fold를 max_epochs까지 돌리고, epoch 1..max 각각의 OOF macroF1을 리포트.
"""
import os, argparse, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
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
def proba(model, loader, device):
    model.eval(); out=[]
    for b in loader:
        b.pop("labels", None)
        b={k:v.to(device) for k,v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo = model(**b).logits.float()
        out.append(torch.softmax(lo,-1).cpu().numpy())
    return np.concatenate(out,0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_epochs", type=int, default=6)   # 1..6 전부 평가
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--num_workers", type=int, default=3)
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()
    print("ARGS:", vars(args)); set_seed(args.seed); device="cuda"

    ids, texts, ys, groups = load_train(args.data_dir)
    ids, texts, ys, groups = np.array(ids), np.array(texts,dtype=object), np.array(ys), np.array(groups)
    from src.data import SPECIAL_TOKENS
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    os.makedirs("oof", exist_ok=True)

    # epoch별 OOF 누적 버퍼
    oof_by_ep = {e: np.zeros((len(ids), NUM_CLASSES), np.float32) for e in range(1, args.max_epochs+1)}

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold,(tr,va) in enumerate(sgkf.split(texts, ys, groups)):
        t0=time.time(); print(f"\n== FOLD {fold} | train {len(tr)} val {len(va)} ==")
        cw=None
        if args.class_weight:
            w=compute_class_weight("balanced", classes=np.unique(ys[tr]), y=ys[tr])
            cw=np.ones(NUM_CLASSES,np.float32)
            for c,wi in zip(np.unique(ys[tr]),w): cw[c]=wi
            cw=torch.tensor(cw,device=device)
        model=AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=NUM_CLASSES, attn_implementation="eager").to(device)
        model.resize_token_embeddings(len(tok))
        trl=DataLoader(DS(texts[tr].tolist(),ys[tr].tolist(),tok,args.max_len),
                       batch_size=args.batch,shuffle=True,num_workers=args.num_workers,
                       pin_memory=True,drop_last=True)
        val=DataLoader(DS(texts[va].tolist(),ys[va].tolist(),tok,args.max_len),
                       batch_size=args.batch*2,shuffle=False,num_workers=args.num_workers,pin_memory=True)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
        steps=len(trl)*args.max_epochs
        sch=get_cosine_schedule_with_warmup(opt,int(steps*args.warmup),steps)
        lossf=torch.nn.CrossEntropyLoss(weight=cw)
        for ep in range(1, args.max_epochs+1):
            model.train()
            for b in trl:
                y=b.pop("labels").to(device); b={k:v.to(device) for k,v in b.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss=lossf(model(**b).logits.float(), y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sch.step(); opt.zero_grad()
            p=proba(model,val,device); oof_by_ep[ep][va]=p
            f1=f1_score(ys[va],p.argmax(1),labels=list(range(NUM_CLASSES)),average="macro",zero_division=0)
            print(f"  ep{ep}: val macroF1={f1:.4f}")
        del model; torch.cuda.empty_cache()
        print(f"  fold {fold} done {time.time()-t0:.0f}s")

    print("\n===== EPOCH SWEEP (OOF macroF1) =====")
    best=(-1,None)
    for ep in range(1, args.max_epochs+1):
        pred=oof_by_ep[ep].argmax(1)
        f1=f1_score(ys,pred,labels=list(range(NUM_CLASSES)),average="macro",zero_division=0)
        acc=(pred==ys).mean()
        print(f"  epoch {ep}: OOF macroF1={f1:.4f} | acc={acc:.4f}")
        np.save(f"oof/{args.tag}_ep{ep}.npy", oof_by_ep[ep])
        if f1>best[0]: best=(f1,ep)
    print(f"\n>>> BEST: epoch {best[1]} -> OOF macroF1={best[0]:.4f}")

if __name__=="__main__":
    main()
