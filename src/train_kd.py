"""앙상블 증류: 4-teacher(EuroBERT+mmBERT계열) OOF soft label로 student(mmBERT-base) 학습.
- loss = (1-a)*CE(hard) + a*KL(student || soft)
- soft label이 OOF라 leakage 없음. fold 분할 seed 42 동일.
"""
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
    def __init__(self, texts, labels, soft, tok, max_len):
        self.texts, self.labels, self.soft, self.tok, self.max_len = texts, labels, soft, tok, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        item["soft"] = torch.tensor(self.soft[i], dtype=torch.float32)
        return item

@torch.inference_mode()
def proba(model, loader, device):
    model.eval(); out=[]
    for b in loader:
        b={k:v.to(device) for k,v in b.items() if k not in ("labels","soft")}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo=model(**b).logits.float()
        out.append(torch.softmax(lo,-1).cpu().numpy())
    return np.concatenate(out,0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--alpha", type=float, default=0.7, help="soft 비중")
    ap.add_argument("--kd_T", type=float, default=2.0, help="증류 temperature")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="kd_student")
    args=ap.parse_args(); print("ARGS:", vars(args))
    set_seed(args.seed); device="cuda"

    ids, texts, ys, groups = load_train("data/original_data")
    texts=np.array(texts,dtype=object); ys=np.array(ys); groups=np.array(groups)
    L=list(range(NUM_CLASSES))

    # 4-teacher OOF 평균 = soft label (EuroBERT 포함)
    soft=(np.load("oof/eurobert5_ep4.npy")+np.load("oof/fields_mmbert_ep3.npy")
          +np.load("oof/soup_lr5_ep3.npy")+np.load("oof/soup_s43_ep3.npy"))/4
    tok=AutoTokenizer.from_pretrained(args.model)
    oof=np.zeros((len(ys),NUM_CLASSES),np.float32)

    sgkf=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    for fold,(tr,va) in enumerate(sgkf.split(texts,ys,groups)):
        t0=time.time(); print(f"\n== FOLD {fold} ==")
        model=AutoModelForSequenceClassification.from_pretrained(
            args.model,num_labels=NUM_CLASSES,attn_implementation="eager").to(device)
        trl=DataLoader(DS(texts[tr].tolist(),ys[tr].tolist(),soft[tr],tok,args.max_len),
                       batch_size=args.batch,shuffle=True,num_workers=3,pin_memory=True,drop_last=True)
        val=DataLoader(DS(texts[va].tolist(),ys[va].tolist(),soft[va],tok,args.max_len),
                       batch_size=args.batch*2,shuffle=False,num_workers=3,pin_memory=True)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
        steps=len(trl)*args.epochs
        sch=get_cosine_schedule_with_warmup(opt,int(steps*0.06),steps)
        T=args.kd_T; a=args.alpha
        for ep in range(1,args.epochs+1):
            model.train()
            for b in trl:
                y=b.pop("labels").to(device); sf=b.pop("soft").to(device)
                b={k:v.to(device) for k,v in b.items()}
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    logits=model(**b).logits.float()
                ce=F.cross_entropy(logits,y)
                kd=F.kl_div(F.log_softmax(logits/T,-1),
                            F.softmax(torch.log(torch.clamp(sf,1e-9,1))/T,-1),
                            reduction="batchmean")*(T*T)
                loss=(1-a)*ce + a*kd
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sch.step(); opt.zero_grad()
            p=proba(model,val,device)
            f1=f1_score(ys[va],p.argmax(1),labels=L,average="macro",zero_division=0)
            print(f"  ep{ep}: val macroF1={f1:.4f}")
        oof[va]=p
        del model; torch.cuda.empty_cache()
        print(f"  fold {fold} done {time.time()-t0:.0f}s")

    f1=f1_score(ys,oof.argmax(1),labels=L,average="macro",zero_division=0)
    np.save(f"oof/{args.tag}.npy",oof)
    print(f"\n>>> KD student OOF macroF1={f1:.4f}  (기준: base 0.7735 / 4-teacher 0.7823)")

if __name__=="__main__":
    main()
