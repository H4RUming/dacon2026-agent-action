"""
Greedy model soup 검증 (fold 단위 OOF).
- fold k: 각 재료의 fold-k 가중치를 val macroF1 순으로 정렬
- 최고부터 시작, 하나씩 평균에 추가 -> val 오르면 유지, 내리면 제외
- 최종: soup OOF macroF1 vs 최고 단일
"""
import os, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data import load_train, NUM_CLASSES

class DS(Dataset):
    def __init__(self, texts, tok, max_len):
        self.texts, self.tok, self.max_len = texts, tok, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); out=[]
    for b in loader:
        b={k:v.to(device) for k,v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo=model(**b).logits.float()
        out.append(torch.softmax(lo,-1).cpu().numpy())
    return np.concatenate(out,0)

def load_sd(path):
    from safetensors.torch import load_file
    return load_file(os.path.join(path,"model.safetensors"))

def avg_sds(sds):
    out={}
    for k in sds[0]:
        out[k]=torch.stack([sd[k].float() for sd in sds],0).mean(0)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="checkpoints/{tag}/fold{k} 재료들")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--batch", type=int, default=128)
    args=ap.parse_args()
    device="cuda"

    ids, texts, ys, groups = load_train("data/original_data")
    texts=np.array(texts,dtype=object); ys=np.array(ys); groups=np.array(groups)
    L=list(range(NUM_CLASSES))
    tok=AutoTokenizer.from_pretrained(args.model)
    base=AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=NUM_CLASSES, attn_implementation="eager").to(device)

    soup_oof=np.zeros((len(ys),NUM_CLASSES),np.float32)
    sgkf=StratifiedGroupKFold(5, shuffle=True, random_state=42)
    for fold,(tr,va) in enumerate(sgkf.split(texts,ys,groups)):
        val_loader=DataLoader(DS(texts[va].tolist(),tok,args.max_len),
                              batch_size=args.batch,num_workers=3,pin_memory=True)
        yv=ys[va]
        def score(sd):
            base.load_state_dict({k:v.to(device) for k,v in sd.items()})
            p=predict(base,val_loader,device)
            return f1_score(yv,p.argmax(1),labels=L,average="macro",zero_division=0), p

        # 각 재료 단독 점수
        cands=[]
        for t in args.tags:
            sd=load_sd(f"checkpoints/{t}/fold{fold}")
            s,_=score(sd)
            cands.append((s,t,sd))
            print(f"  fold{fold} {t}: {s:.4f}")
        cands.sort(key=lambda x:-x[0])

        # greedy
        soup=[cands[0][2]]; names=[cands[0][1]]; best_s,best_p=score(avg_sds(soup))
        for s,t,sd in cands[1:]:
            trial=avg_sds(soup+[sd])
            ts,tp=score(trial)
            if ts>best_s:
                soup.append(sd); names.append(t); best_s,best_p=ts,tp
                print(f"  fold{fold} + {t} -> {ts:.4f} (유지)")
            else:
                print(f"  fold{fold} + {t} -> {ts:.4f} (제외)")
        soup_oof[va]=best_p
        print(f"== fold{fold} soup={names} macroF1={best_s:.4f}")

    mf=f1_score(ys,soup_oof.argmax(1),labels=L,average="macro",zero_division=0)
    np.save("oof/greedy_soup.npy",soup_oof)
    print(f"\n>>> GREEDY SOUP OOF macroF1={mf:.4f}")

if __name__=="__main__":
    main()
