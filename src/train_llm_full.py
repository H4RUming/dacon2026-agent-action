"""Gemma4 E2B 텍스트백본 QLoRA 분류 파인튜닝 (fold0 판정용)"""
import os, argparse, random, time
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import sys; sys.path.insert(0, ".")
from src.data import load_train, NUM_CLASSES

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class DS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts,self.labels,self.tok,self.max_len=texts,labels,tok,max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self,i):
        enc=self.tok(self.texts[i],truncation=True,max_length=self.max_len,
                     padding="max_length",return_tensors="pt")
        return {"input_ids":enc["input_ids"].squeeze(0),
                "attention_mask":enc["attention_mask"].squeeze(0),
                "labels":torch.tensor(self.labels[i])}

class LLMClassifier(nn.Module):
    def __init__(self, backbone, hidden, ncls):
        super().__init__()
        self.backbone=backbone
        self.head=nn.Linear(hidden, ncls)
    def forward(self, input_ids, attention_mask):
        out=self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        h=out.hidden_states[-1]
        # 마지막 유효 토큰 위치 풀링
        # padding 방향 무관: 유효 토큰의 마지막 위치
        idx=(attention_mask * torch.arange(attention_mask.size(1), device=h.device)).argmax(1)
        pooled=h[torch.arange(h.size(0),device=h.device), idx]
        return self.head(pooled.float())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-e2b-it")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=576)
    ap.add_argument("--folds_run", type=int, default=1, help="앞에서부터 N개 fold만")
    ap.add_argument("--tag", default="llm_e2b")
    args=ap.parse_args(); print("ARGS:", vars(args))
    set_seed(42); device="cuda"

    ids,texts,ys,groups=load_train("data/original_data")
    texts=np.array(texts,dtype=object); ys=np.array(ys); groups=np.array(groups)
    L=list(range(NUM_CLASSES))
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    oof=np.zeros((len(ys),NUM_CLASSES),np.float32)
    sgkf=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    for fold,(tr,va) in enumerate(sgkf.split(texts,ys,groups)):
        if fold>=args.folds_run: break
        t0=time.time(); print(f"\n== FOLD {fold} ==")
        full=AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        backbone=full   # 전체 모델 그대로 (Gemma4 cross-layer KV 공유 때문에 서브모듈 분리 불가)
        print(f"params: {sum(p.numel() for p in backbone.parameters())/1e9:.2f}B")
        hidden=full.config.text_config.hidden_size if hasattr(full.config,"text_config") else full.config.hidden_size
        model=LLMClassifier(backbone, hidden, NUM_CLASSES).to(device)

        trl=DataLoader(DS(texts[tr].tolist(),ys[tr].tolist(),tok,args.max_len),
                       batch_size=args.batch,shuffle=True,num_workers=3,pin_memory=True,drop_last=True)
        val=DataLoader(DS(texts[va].tolist(),ys[va].tolist(),tok,args.max_len),
                       batch_size=args.batch*2,shuffle=False,num_workers=3,pin_memory=True)
        params=[p for p in model.parameters() if p.requires_grad]
        print(f"학습 파라미터: {sum(p.numel() for p in params)/1e6:.1f}M")
        opt=torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.lr*10},
        ], weight_decay=0.01)
        steps=len(trl)//args.grad_accum*args.epochs
        sch=get_cosine_schedule_with_warmup(opt,int(steps*0.06),steps)
        for ep in range(1,args.epochs+1):
            model.train(); step=0
            for b in trl:
                y=b.pop("labels").to(device)
                b={k:v.to(device) for k,v in b.items()}
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                    logits=model(**b)
                loss=F.cross_entropy(logits,y)/args.grad_accum
                loss.backward(); step+=1
                if step%args.grad_accum==0:
                    torch.nn.utils.clip_grad_norm_(params,1.0)
                    opt.step(); sch.step(); opt.zero_grad()
            # val
            model.eval(); out=[]
            with torch.inference_mode():
                for b in val:
                    b.pop("labels")
                    b={k:v.to(device) for k,v in b.items()}
                    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                        lo=model(**b)
                    out.append(torch.softmax(lo,-1).float().cpu().numpy())
            p=np.concatenate(out,0)
            f1=f1_score(ys[va],p.argmax(1),labels=L,average="macro",zero_division=0)
            print(f"  ep{ep}: val macroF1={f1:.4f} ({time.time()-t0:.0f}s)")
        oof[va]=p
        os.makedirs(f"checkpoints/{args.tag}",exist_ok=True)
        backbone.save_pretrained(f"checkpoints/{args.tag}/fold{fold}_lora")
        torch.save(model.head.state_dict(), f"checkpoints/{args.tag}/fold{fold}_head.pt")
        del model, full, backbone; torch.cuda.empty_cache()
    np.save(f"oof/{args.tag}_partial.npy",oof)
    print("\ndone")

if __name__=="__main__":
    main()
