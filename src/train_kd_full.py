"""KD 전량학습: 70k 전체 + OOF soft label -> submit/model"""
import os, argparse, random
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from src.data import load_train, NUM_CLASSES

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class DS(Dataset):
    def __init__(self, texts, labels, soft, tok, max_len):
        self.texts,self.labels,self.soft,self.tok,self.max_len=texts,labels,soft,tok,max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc=self.tok(self.texts[i],truncation=True,max_length=self.max_len,
                     padding="max_length",return_tensors="pt")
        item={k:v.squeeze(0) for k,v in enc.items()}
        item["labels"]=torch.tensor(self.labels[i],dtype=torch.long)
        item["soft"]=torch.tensor(self.soft[i],dtype=torch.float32)
        return item

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--kd_T", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="submit/model_kd")
    args=ap.parse_args(); print("ARGS:", vars(args))
    set_seed(args.seed); device="cuda"

    ids, texts, ys, groups = load_train("data/original_data")
    soft=(np.load("oof/eurobert5_ep4.npy")+np.load("oof/fields_mmbert_ep3.npy")
          +np.load("oof/soup_lr5_ep3.npy")+np.load("oof/soup_s43_ep3.npy"))/4
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForSequenceClassification.from_pretrained(
        args.model,num_labels=NUM_CLASSES,attn_implementation="eager").to(device)
    dl=DataLoader(DS(list(texts),list(ys),soft,tok,args.max_len),
                  batch_size=args.batch,shuffle=True,num_workers=3,pin_memory=True,drop_last=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    steps=len(dl)*args.epochs
    sch=get_cosine_schedule_with_warmup(opt,int(steps*0.06),steps)
    T=args.kd_T; a=args.alpha
    import time
    for ep in range(args.epochs):
        model.train(); t0=time.time(); tot=0.0; nb=0
        for b in dl:
            y=b.pop("labels").to(device); sf=b.pop("soft").to(device)
            b={k:v.to(device) for k,v in b.items()}
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                logits=model(**b).logits.float()
            ce=F.cross_entropy(logits,y)
            kd=F.kl_div(F.log_softmax(logits/T,-1),
                        F.softmax(torch.log(torch.clamp(sf,1e-9,1))/T,-1),
                        reduction="batchmean")*(T*T)
            loss=(1-a)*ce+a*kd
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tot+=loss.item(); nb+=1
        print(f"  epoch {ep}: loss={tot/nb:.4f} | {time.time()-t0:.0f}s")
    os.makedirs(args.out,exist_ok=True)
    model.half().save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")

if __name__=="__main__":
    main()
