"""
bias 최소-노름 정규화: wF1이 유지되는 한 각 좌표를 0 방향으로 탐욕 축소.
(OOF에서 비용 없는 평평한 방향의 큰 bias는 test에서 위험 — 같은 목적값의 보수적 해를 선택)

사용: python src/shrink_bias.py <oof1.npy> <oof2.npy> ... --bias <in.npy> -o <out.npy>
결정론적 (RNG 부재). 제출 #17의 bias 재생성 절차의 2단계.
"""
import argparse
import numpy as np
import pandas as pd

REPO = "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oofs", nargs="+")
    ap.add_argument("--bias", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()

    meta = pd.read_parquet(f"{REPO}/oof/mmbert_base_meta.parquet")
    y = meta["label"].values.astype(int)
    sess = meta["id"].str.rsplit("-step_", n=1).str[0].values
    _, sess_inv, counts = np.unique(sess, return_inverse=True, return_counts=True)
    w = 1.0 / counts[sess_inv]
    P = sum(np.load(p) for p in args.oofs) / len(args.oofs)
    logp = np.log(np.clip(P, 1e-9, None))

    def wf1(b):
        pred = np.argmax(logp + b, axis=1)
        f = []
        for c in range(14):
            tp = w[(pred == c) & (y == c)].sum()
            fp = w[(pred == c) & (y != c)].sum()
            fn = w[(pred != c) & (y == c)].sum()
            f.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
        return float(np.mean(f))

    b = np.load(args.bias).copy()
    best = wf1(b)
    for _ in range(args.passes):
        for c in range(14):
            while abs(b[c]) > 1e-9:
                trial = b.copy()
                trial[c] = np.round(trial[c] - np.sign(trial[c]) * args.step, 3)
                if wf1(trial) >= best - 1e-9:
                    b = trial
                else:
                    break
    print(f"shrunk wF1={wf1(b):.5f} (start {best:.5f})")
    print("bias:", np.round(b, 3).tolist())
    np.save(args.out, b)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
