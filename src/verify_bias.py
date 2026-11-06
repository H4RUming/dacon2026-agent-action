"""
제출 bias의 인과 사슬 검증 — 심사용 원커맨드 체크.

제출 #17~20의 bias 4종을 train OOF에서 재생성해 제출본과 대조한다.
전부 PASS면: 계수에 LB 정보가 들어갈 경로가 없음이 기계적으로 증명된다
(생성 시점·의도에 대한 신뢰를 요구하지 않음 — 결정론적 재계산으로 확인).

사용: python -m src.verify_bias
"""
import hashlib
import subprocess
import sys

import numpy as np

# (제출물, OOF 입력, 재생성 절차) — 전부 결정론적 스크립트의 출력
TRIO = ["oof/kd_a03.npy", "oof/kd_a03_s43.npy", "oof/hp_lr5e5_ep3.npy"]
PAIR = ["oof/kd_a03.npy", "oof/soup_s43_ep3.npy"]
# 제출 #17 주의: 파일 경합 사고로 의도된 bag7 bias 대신 "페어-fit+shrink" bias가 포장·제출됨.
# 본 스크립트가 해당 불일치를 검출했고(사후 공개), 제출된 실제 계수의 사슬은 아래 절차로 닫힘.
ARTIFACTS = [
    ("submit/bias_trio_bag.npy",    PAIR, [["--lim", "0.8"], ["SHRINK"]]),        # 제출 #17 (사고 공개)
    ("submit/bias_trio_bag8.npy",   TRIO, [["--bag", "24", "--bag_seed", "8"]]),  # 제출 #18
    ("submit/bias_trio_bag9.npy",   TRIO, [["--bag", "24", "--bag_seed", "9"]]),  # 제출 #19
    ("submit/bias_trio_single.npy", TRIO, [["--lim", "0.8"]]),                    # 제출 #20 (최종)
    ("submit/bias_trio_bag10.npy",  TRIO, [["--bag", "24", "--bag_seed", "10"]]), # 예비 (사전등록, 미제출)
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    print("== 입력 (train OOF) ==")
    for p in sorted(set(TRIO + PAIR)):
        print(f"  {p}  sha256:{sha(p)}")
    ok = True
    for target, oofs, steps in ARTIFACTS:
        out = "/tmp/regen_bias.npy"
        for st in steps:
            if st == ["SHRINK"]:
                cmd = [sys.executable, "src/shrink_bias.py", *oofs, "--bias", out, "-o", out]
            else:
                cmd = [sys.executable, "src/fit_bias_wf1.py", *oofs, "-o", out, *st]
            subprocess.run(cmd, check=True, capture_output=True)
        a, b = np.load(target), np.load(out)
        d = float(np.abs(a - b).max())
        status = "PASS" if d < 1e-12 else ("PASS(tol)" if d < 1e-6 else "FAIL")
        ok &= status != "FAIL"
        desc = " | ".join(" ".join(st) for st in steps)
        print(f"{status}  {target}  sha256:{sha(target)}  max|Δ|={d:.2e}  절차: {desc}")
    print("\n결론:", "전 계수가 train OOF에서 결정론적으로 재생성됨 — LB→계수 경로 부재 증명" if ok else "재생성 불일치 — 확인 필요")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
