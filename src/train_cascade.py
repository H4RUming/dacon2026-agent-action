"""4-class 전용 판별기 (read/grep/list/glob) 5-fold OOF.
fold 분할은 전체 데이터 기준 seed 42와 동일하게 유지 (1단계 OOF와 정렬)."""
import os, argparse, random, time
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from src.data import load_train, CLASSES

SUB = ["read_file","grep_search","list_directory","glob_pattern"]
SUB_IDS = [CLASSES.index(c) for c in SUB]

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
            lo=model(**b).logits.float()
        out.append(torch.softmax(lo,-1).cpu().numpy())
    return np.concatenate(out,0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--tag", default="cascade4")
    args=ap.parse_args(); print("ARGS:", vars(args))
    set_seed(42); device="cuda"

    ids, texts, ys, groups = load_train("data/original_data")
    texts=np.array(texts,dtype=object); ys=np.array(ys); groups=np.array(groups)
    tok=AutoTokenizer.from_pretrained(args.model)

    # 전체 기준 fold 분할 (1단계와 동일) -> 그 안에서 4클래스만 사용
    fold_arr=np.full(len(ys),-1)
    for f,(tr,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(texts,ys,groups)):
        fold_arr[va]=f
    sub_mask=np.isin(ys, SUB_IDS)
    remap={g:i for i,g in enumerate(SUB_IDS)}
    ys4=np.array([remap.get(y,-1) for y in ys])
    print(f"4-class rows: {sub_mask.sum()}")

    oof4=np.zeros((len(ys),4),np.float32)
    os.makedirs(f"checkpoints/{args.tag}",exist_ok=True)
    for f in range(5):
        tr=np.where(sub_mask&(fold_arr!=f))[0]; va=np.where(sub_mask&(fold_arr==f))[0]
        print(f"\n== FOLD {f} | train {len(tr)} val {len(va)} ==")
        model=AutoModelForSequenceClassification.from_pretrained(
            args.model,num_labels=4,attn_implementation="eager").to(device)
        trl=DataLoader(DS(texts[tr].tolist(),ys4[tr].tolist(),tok,args.max_len),
                       batch_size=args.batch,shuffle=True,num_workers=3,pin_memory=True,drop_last=True)
        val=DataLoader(DS(texts[va].tolist(),ys4[va].tolist(),tok,args.max_len),
                       batch_size=args.batch*2,shuffle=False,num_workers=3,pin_memory=True)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
        steps=len(trl)*args.epochs
        sch=get_cosine_schedule_with_warmup(opt,int(steps*0.06),steps)
        lossf=torch.nn.CrossEntropyLoss()
        for ep in range(args.epochs):
            model.train(); t0=time.time()
            for b in trl:
                y=b.pop("labels").to(device); b={k:v.to(device) for k,v in b.items()}
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    loss=lossf(model(**b).logits.float(),y)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); sch.step(); opt.zero_grad()
            p=proba(model,val,device)
            f1=f1_score(ys4[va],p.argmax(1),labels=list(range(4)),average="macro",zero_division=0)
            print(f"  ep{ep+1}: 4class macroF1={f1:.4f} ({time.time()-t0:.0f}s)")
        oof4[va]=p
        fd=f"checkpoints/{args.tag}/fold{f}"; os.makedirs(fd,exist_ok=True)
        model.half().save_pretrained(fd)
        del model; torch.cuda.empty_cache()

    np.save(f"oof/{args.tag}.npy",oof4)
    m=sub_mask
    f1=f1_score(ys4[m],oof4[m].argmax(1),labels=list(range(4)),average="macro",zero_division=0)
    print(f"\n>>> 4-class 전용 OOF macroF1={f1:.4f}")
    print("(비교용: 1단계 앙상블의 같은 4클래스 서브셋 성능은 별도 스크립트로)")

if __name__=="__main__":
    main()
