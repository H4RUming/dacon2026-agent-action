"""
제출용 추론 (T4, 오프라인). data/test.jsonl -> output/submission.csv
- serialize_example()은 학습(src/data.py 문맥전체)과 반드시 동일
- 길이정렬 배치로 패딩 낭비 최소화 (T4 10분 준수)
"""
import os, json, csv, glob, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
MAX_LEN = 640
BATCH = 256

CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

def load_bias() -> np.ndarray:
    path = os.path.join(MODEL_DIR, "bias.npy")
    if not os.path.exists(path):
        return np.zeros(14, dtype=np.float32)
    bias = np.load(path).astype(np.float32)
    if bias.shape != (14,):
        raise ValueError(f"bias.npy shape must be (14,), got {bias.shape}")
    return bias

BIAS = load_bias()

# ===== 학습(src/data.py)과 동일한 문맥전체 직렬화 =====
def serialize_example(sample: dict, n_recent_actions: int = 999) -> str:
    """문맥 전체 버전: history 전량 + 모든 args + result. (max_len으로 자연 truncate)"""
    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    hist = sample.get("history", []) or []

    parts = []
    parts.append("[PROMPT] " + (sample.get("current_prompt", "") or ""))
    # 코드베이스 주언어 (상위 2개만, 토큰 절약)
    lm = ws.get("language_mix", {}) or {}
    top_langs = ",".join(k for k, _ in sorted(lm.items(), key=lambda x: -x[1])[:2])
    # current_prompt가 가리키는 파일이 열려있거나 history에 있었나
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
    # 열린 파일 목록도 (경로가 read/edit 신호일 수 있음)
    of = ws.get("open_files", []) or []
    if of:
        parts.append("[OPEN] " + ", ".join(of))
    # history 전체를 시간순으로
    parts.append("[HISTORY]")
    for h in hist:
        if h.get("role") == "user":
            parts.append("U: " + (h.get("content", "") or ""))
        else:
            args = h.get("args", {}) or {}
            arg_str = " ".join(f"{k}={v}" for k, v in args.items())
            parts.append(f"A: {h.get('name','')}({arg_str}) -> {h.get('result_summary','') or ''}")
    return "\n".join(parts)

def find_test_path():
    for p in ["data/test.jsonl", "data/original_data/test.jsonl",
              os.path.join(HERE, "data/test.jsonl")]:
        if os.path.exists(p):
            return p
    hits = glob.glob("**/test.jsonl", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("test.jsonl not found under data/")

@torch.inference_mode()
def predict_indices(model, tok, ser, idxs, device, bias_t, batch_size):
    if not idxs:
        return {}
    chunk = [ser[i] for i in idxs]
    try:
        enc = tok(chunk, truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.float()
        logits = logits + bias_t.to(device=logits.device, dtype=logits.dtype)
        pr = logits.argmax(-1).cpu().numpy()
        return {ii: int(pr[j]) for j, ii in enumerate(idxs)}
    except Exception as e:
        if batch_size <= 1 or len(idxs) <= 1:
            sys.stderr.write(f"[warn] single sample {idxs[0]} failed: {e}; respond_only fallback\n")
            return {}
        next_size = max(1, batch_size // 2)
        sys.stderr.write(f"[warn] batch size {batch_size} failed: {e}; retrying with {next_size}\n")
        out = {}
        for start in range(0, len(idxs), next_size):
            out.update(predict_indices(model, tok, ser, idxs[start:start + next_size],
                                       device, bias_t, next_size))
        return out

def main():
    test_path = find_test_path()
    os.makedirs("output", exist_ok=True)
    out_path = "output/submission.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, attn_implementation="eager"
    ).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    samples = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    ids = [s.get("id", f"row_{i}") for i, s in enumerate(samples)]
    n = len(samples)
    preds = ["respond_only"] * n   # 폴백 기본값 (크래시 대비)
    bias_t = torch.tensor(BIAS, device=device)

    # 직렬화 + 길이정렬 (짧은 것부터 -> 배치 패딩 낭비 최소화)
    ser = [serialize_example(s) for s in samples]
    try:
        lengths = [len(tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]) for t in ser]
        order = list(np.argsort(lengths))
    except Exception:
        order = list(range(n))   # 정렬 실패해도 그냥 원순서로 진행

    for start in range(0, n, BATCH):
        idxs = order[start:start + BATCH]
        got = predict_indices(model, tok, ser, idxs, device, bias_t, len(idxs))
        for ii, pred_idx in got.items():
            preds[ii] = CLASSES[pred_idx]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "action"])
        for i, p in zip(ids, preds):
            w.writerow([i, p])
    print(f"wrote {out_path} | {n} rows")

if __name__ == "__main__":
    main()
