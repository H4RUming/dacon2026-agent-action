"""
70k 전체로 mmBERT-base 재학습 (제출용 단일 모델).
- fold 분할 없음, 전량 학습
- val 없으니 epoch 수는 5-fold에서 검증된 값 사용 (기본 3, 필요시 4)
- fp16으로 submit/model/ 에 저장 (config+tokenizer 포함)
"""
import os, argparse, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_data")
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--out", default="submit/model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=640)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--num_workers", type=int, default=3)
    args = ap.parse_args()
    print("ARGS:", vars(args))
    set_seed(args.seed)
    device = "cuda"

    ids, texts, ys, groups = load_train(args.data_dir)
    print(f"loaded {len(ids)} rows | classes={NUM_CLASSES}")

    tok = AutoTokenizer.from_pretrained(args.model)
    cw = None
    if args.class_weight:
        w = compute_class_weight("balanced", classes=np.unique(ys), y=np.array(ys))
        cw = np.ones(NUM_CLASSES, dtype=np.float32)
        for c, wi in zip(np.unique(ys), w): cw[c] = wi
        cw = torch.tensor(cw, device=device)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=NUM_CLASSES, attn_implementation="eager"
    ).to(device)

    loader = DataLoader(DS(texts, ys, tok, args.max_len), batch_size=args.batch,
                        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps*args.warmup), total_steps)
    lossf = torch.nn.CrossEntropyLoss(weight=cw)

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); running = 0.0
        for step, batch in enumerate(loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
                loss = lossf(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
        print(f"  epoch {ep}: loss={running/len(loader):.4f} | {time.time()-t0:.0f}s")

    os.makedirs(args.out, exist_ok=True)
    model.half().save_pretrained(args.out); tok.save_pretrained(args.out)
    # 파일 크기 확인
    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)
                if os.path.isfile(os.path.join(args.out, f)))
    print(f"\nsaved -> {args.out} | total {total/1e6:.0f} MB")
    print("files:", sorted(os.listdir(args.out)))

if __name__ == "__main__":
    main()
