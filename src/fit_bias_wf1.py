"""
정확-기댓값 세션-균등 bias 피팅 (su2 몬테카를로 프로토콜의 결정론적 대체).

세션-균등 test 분포에서 macro-F1의 기댓값은 "행 가중치 1/세션길이"의
가중 confusion count로 닫힌 형태로 계산된다. 기존 su2 프로토콜(10-seed
몬테카를로 서브샘플 coordinate ascent)은 seed당 F1 표준편차 ~0.004의
노이즈 위에서 argmax를 찾았으므로, 이 스크립트는 같은 목적함수의
기댓값을 직접 최적화한다 (grid 0.05, 3 sweeps).

사용: python src/fit_bias_wf1.py <oof1.npy> <oof2.npy> ... -o submit/bias_X.npy
모든 계수는 train OOF에서만 유도된다 (LB 피드백 불사용).
"""
import argparse
import numpy as np
import pandas as pd

REPO = "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oofs", nargs="+", help="oof .npy paths (equal-weight prob average)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--grid", type=float, default=0.05)
    ap.add_argument("--lim", type=float, default=0.6)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--bag_seed", type=int, default=7)
    ap.add_argument("--bag", type=int, default=0,
                    help="세션-부트스트랩 B회 피팅 후 평균 (임계값 분산 감소). 0=단일 피팅")
    args = ap.parse_args()

    meta = pd.read_parquet(f"{REPO}/oof/mmbert_base_meta.parquet")
    y = meta["label"].values.astype(int)
    sess = meta["id"].str.rsplit("-step_", n=1).str[0].values
    _, sess_inv, counts = np.unique(sess, return_inverse=True, return_counts=True)
    w = 1.0 / counts[sess_inv]  # 세션-균등 기대 가중치

    P = sum(np.load(p) for p in args.oofs) / len(args.oofs)
    logp = np.log(np.clip(P, 1e-9, None))

    def wf1(bias):
        pred = np.argmax(logp + bias, axis=1)
        f1s = []
        for c in range(14):
            tp = w[(pred == c) & (y == c)].sum()
            fp = w[(pred == c) & (y != c)].sum()
            fn = w[(pred != c) & (y == c)].sum()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
        return float(np.mean(f1s))

    grid = np.round(np.arange(-args.lim, args.lim + 1e-9, args.grid), 3)

    def wf1_rows(bias, rows):
        pred = np.argmax(logp[rows] + bias, axis=1)
        yy, ww = y[rows], w[rows]
        f1s = []
        for c in range(14):
            tp = ww[(pred == c) & (yy == c)].sum()
            fp = ww[(pred == c) & (yy != c)].sum()
            fn = ww[(pred != c) & (yy == c)].sum()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
        return float(np.mean(f1s))

    def coord_ascent(rows, sweeps):
        b = np.zeros(14)
        best = wf1_rows(b, rows)
        for _ in range(sweeps):
            for c in range(14):
                cur = b[c]
                for g in grid:
                    b[c] = g
                    s = wf1_rows(b, rows)
                    if s > best + 1e-12:
                        best, cur = s, g
                b[c] = cur
        return b, best

    all_rows = np.arange(len(y))
    if args.bag > 0:
        # 세션-레벨 부트스트랩 B회 → bias 평균 (분산 감소)
        rng = np.random.RandomState(args.bag_seed)
        n_sess = sess_inv.max() + 1
        row_by_sess = [[] for _ in range(n_sess)]
        for r in range(len(y)):
            row_by_sess[sess_inv[r]].append(r)
        biases = []
        for k in range(args.bag):
            bs = rng.randint(0, n_sess, size=n_sess)
            rows = np.concatenate([row_by_sess[s] for s in bs])
            bk, sk = coord_ascent(rows, sweeps=1)
            biases.append(bk)
            print(f"bag {k+1}/{args.bag}: wF1={sk:.5f}")
        b = np.mean(biases, 0)
        print(f"bagged bias wF1 (full data): {wf1_rows(b, all_rows):.5f}")
    else:
        b, best = coord_ascent(all_rows, args.sweeps)
        print(f"final wF1={best:.5f}")
    np.save(args.out, b)
    print("bias:", np.round(b, 3).tolist())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
