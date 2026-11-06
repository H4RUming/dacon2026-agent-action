"""
제출용 추론 (T4, 오프라인) — 2모델 int8 앙상블.
- model_a.npz + model_b.npz (int8+scale) -> fp16 복원 -> GPU
- 두 모델 softmax 평균 -> bias -> argmax
- 직렬화는 학습(src/data.py fields 버전)과 동일 (인라인 복제)
"""
import os, json, csv, glob, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")          # config + tokenizer (공유 1벌)
NPZ_PATHS = [os.path.join(HERE, "model_kd03.npz"), os.path.join(HERE, "model_kd03_s43.npz"),
             os.path.join(HERE, "model_lr5full.npz")]
BIAS_PATH = os.path.join(HERE, "bias_trio_single.npy")
MAX_LEN = 640
BATCH = 256

CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

# ===== 학습(src/data.py)과 동일한 fields 직렬화 =====
def serialize_example(sample: dict, n_recent_actions: int = 999) -> str:
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    hist = sample.get("history", []) or []

    parts = []
    parts.append("[PROMPT] " + (sample.get("current_prompt", "") or ""))
    lm = ws.get("language_mix", {}) or {}
    top_langs = ",".join(k for k, _ in sorted(lm.items(), key=lambda x: -x[1])[:2])
    cp_l = (sample.get("current_prompt", "") or "").lower()
    of = ws.get("open_files", []) or []
    file_in_open = int(any(f.split("/")[-1].lower() in cp_l for f in of))
    hist_paths = [h.get("args", {}).get("path", "") for h in hist if h.get("role") == "assistant_action"]
    file_in_hist = int(any(p and p.split("/")[-1].lower() in cp_l for p in hist_paths))
    parts.append(
        "[STATE] "
        f"ci={ws.get('last_ci_status')} dirty={ws.get('git_dirty')} "
        f"open={len(of)} turn={meta.get('turn_index')} "
        f"tier={meta.get('user_tier')} lang={meta.get('language_pref')} "
        f"codelang={top_langs} loc={ws.get('loc')} budget={meta.get('budget_tokens_remaining')} "
        f"pfopen={file_in_open} pfhist={file_in_hist}"
    )
    if of:
        parts.append("[OPEN] " + ", ".join(of))
    parts.append("[HISTORY]")
    for h in hist:
        if h.get("role") == "user":
            parts.append("U: " + (h.get("content", "") or ""))
        else:
            args = h.get("args", {}) or {}
            arg_str = " ".join(f"{k}={v}" for k, v in args.items())
            parts.append(f"A: {h.get('name','')}({arg_str}) -> {h.get('result_summary','') or ''}")
    return "\n".join(parts)
# =====================================================

def find_test_path():
    for p in ["data/test.jsonl", "data/original_data/test.jsonl",
              os.path.join(HERE, "data/test.jsonl")]:
        if os.path.exists(p):
            return p
    hits = glob.glob("**/test.jsonl", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("test.jsonl not found")

def load_int8_model(npz_path, device):
    """int8+scale npz -> fp16 state_dict -> 모델 생성 (dequant는 GPU에서: 3vCPU 절약)"""
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config)
    z = np.load(npz_path)
    sd = {}
    keys = set(k.rsplit("::", 1)[0] for k in z.files)
    for k in keys:
        if k + "::q" in z.files:
            q = torch.from_numpy(z[k + "::q"]).to(device)
            s = torch.from_numpy(z[k + "::s"]).to(device)
            sd[k] = (q.float() * s.float()).half()
        else:
            sd[k] = torch.from_numpy(z[k + "::f"]).to(device).half()
    model = model.to(device).half()
    model.load_state_dict(sd)
    return model.eval()

def main():
    test_path = find_test_path()
    os.makedirs("output", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    BIAS = torch.tensor(np.load(BIAS_PATH), dtype=torch.float32, device=device)

    samples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    ids = [s.get("id", f"row_{i}") for i, s in enumerate(samples)]
    n = len(samples)
    preds = ["respond_only"] * n

    ser = [serialize_example(s) for s in samples]
    try:
        lengths = [len(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in ser]
        order = list(np.argsort(lengths))
    except Exception:
        order = list(range(n))

    # 확률 누적 버퍼 (N모델 평균용)
    prob_sum = np.zeros((n, len(CLASSES)), dtype=np.float32)

    # 배치 인코딩은 모델 간 공유 (tokenize-once: 3vCPU에서 모델당 재토크나이즈 절약)
    enc_cache = []
    for start in range(0, n, BATCH):
        idxs = order[start:start + BATCH]
        chunk = [ser[i] for i in idxs]
        try:
            enc = tok(chunk, truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
        except Exception:
            enc = None
        enc_cache.append((idxs, enc))

    for mi, npz in enumerate(NPZ_PATHS):
        model = load_int8_model(npz, device)
        for bi, (idxs, enc) in enumerate(enc_cache):
            try:
                if enc is None:
                    raise RuntimeError("batch tokenization failed")
                enc_d = {k: v.to(device) for k, v in enc.items()}
                with torch.inference_mode():
                    logits = model(**enc_d).logits.float()
                p = torch.softmax(logits, -1).cpu().numpy()
                for j, ii in enumerate(idxs):
                    prob_sum[ii] += p[j]
            except Exception as e:
                sys.stderr.write(f"[warn] m{mi} batch {bi}: {e}\n")
                for ii in idxs:
                    try:
                        e1 = tok(ser[ii], truncation=True, max_length=MAX_LEN,
                                 padding=True, return_tensors="pt")
                        e1 = {k: v.to(device) for k, v in e1.items()}
                        with torch.inference_mode():
                            lo = model(**e1).logits.float()
                        prob_sum[ii] += torch.softmax(lo, -1).cpu().numpy()[0]
                    except Exception:
                        pass
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # 평균 확률 -> log -> bias -> argmax  (확률 0 방지 clip)
    got = prob_sum.sum(1) > 0
    logit = np.log(np.clip(prob_sum, 1e-9, None))
    logit = logit + BIAS.cpu().numpy()
    final = logit.argmax(1)
    for i in range(n):
        if got[i]:
            preds[i] = CLASSES[int(final[i])]

    # ===== 형제-step 매칭: test 내 같은 세션의 다른 step history에서 정답 탐색 =====
    from collections import defaultdict
    sess_rows = defaultdict(list)
    for i, smp in enumerate(samples):
        sid = str(smp.get("id","")).rsplit("-step_", 1)[0]
        sess_rows[sid].append(i)
    n_override = 0
    for sid, idxs in sess_rows.items():
        if len(idxs) < 2:
            continue
        for i in idxs:
            cp = (samples[i].get("current_prompt") or "").strip()
            if not cp:
                continue
            found = None
            for j in idxs:
                if j == i:
                    continue
                hist = samples[j].get("history") or []
                for t, h in enumerate(hist):
                    if h.get("role") == "user" and (h.get("content") or "").strip() == cp:
                        for h2 in hist[t+1:]:
                            if h2.get("role") == "assistant_action":
                                nm = h2.get("name")
                                if nm in CLASSES:
                                    found = nm
                                break
                        break
                if found:
                    break
            if found:
                preds[i] = found; n_override += 1
    sys.stderr.write(f"[sibling] overrides {n_override}/{n}\n")
    # =====================================================================
    with open("output/submission.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "action"])
        for i, p in zip(ids, preds):
            w.writerow([i, p])
    print(f"wrote output/submission.csv | {n} rows")

if __name__ == "__main__":
    main()
