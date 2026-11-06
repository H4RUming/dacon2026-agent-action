"""
서빙 중인 Gemma 4로 few-shot 체급 판정 (파인튜닝 X).
문맥전체 직렬화 입력 + guided_choice(14-class 강제) -> macro-F1.
"""
import json, argparse, random, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from sklearn.metrics import f1_score, classification_report
from src.data import load_train, CLASSES, CLASS2ID, NUM_CLASSES

client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")
MODEL = "gemma-4-31b-it"

SYS = (
    "You are an AI coding agent. Given the current workspace state, the conversation "
    "history, and the user's latest message, predict the SINGLE next action the agent "
    "will take. Choose exactly one of these 14 actions:\n"
    + ", ".join(CLASSES) + "."
)

def build_messages(shots, x):
    msgs = [{"role": "system", "content": SYS}]
    for ex_text, ex_label in shots:
        msgs.append({"role": "user", "content": ex_text})
        msgs.append({"role": "assistant", "content": ex_label})
    msgs.append({"role": "user", "content": x})
    return msgs

def predict_one(args):
    shots, text = args
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=build_messages(shots, text),
                max_tokens=8, temperature=0.0,
                extra_body={"guided_choice": CLASSES},
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(1)
    return "respond_only"  # 폴백

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--n_shots", type=int, default=14)   # 클래스당 1개
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    ids, texts, ys, groups = load_train("data/original_data")
    ids, texts, ys = np.array(ids), np.array(texts, dtype=object), np.array(ys)

    # few-shot 예시: 클래스별 per_class개씩
    per_class = max(1, args.n_shots // NUM_CLASSES)
    shot_idx = []
    for c in range(NUM_CLASSES):
        cand = np.where(ys == c)[0]
        shot_idx.extend(int(i) for i in cand[:per_class])
    shot_set = set(shot_idx)
    shots = [(texts[i], CLASSES[ys[i]]) for i in shot_idx]
    print(f"few-shot: {len(shots)} examples (클래스별 1개)")

    # 평가셋: shot 제외하고 랜덤 n_eval
    pool = [i for i in range(len(texts)) if i not in shot_set]
    eval_idx = np.random.choice(pool, min(args.n_eval, len(pool)), replace=False)
    eval_texts = [texts[i] for i in eval_idx]
    eval_y = ys[eval_idx]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        preds = list(ex.map(predict_one, [(shots, t) for t in eval_texts]))
    dt = time.time() - t0

    pred_ids = np.array([CLASS2ID.get(p, CLASS2ID["respond_only"]) for p in preds])
    macro = f1_score(eval_y, pred_ids, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    acc = (pred_ids == eval_y).mean()
    print(f"\n===== Gemma few-shot ({len(eval_idx)} samples, {args.n_shots}-shot) =====")
    print(f"macroF1={macro:.4f} | acc={acc:.4f} | {dt:.0f}s ({len(eval_idx)/dt:.1f}/s)")
    print(classification_report(eval_y, pred_ids, labels=list(range(NUM_CLASSES)),
                                target_names=CLASSES, digits=3, zero_division=0))

if __name__ == "__main__":
    main()
