"""fp16 safetensors -> int8+scale npz (디스크 전용, 로드시 fp16 복원)"""
import sys, os, numpy as np, torch
from safetensors.torch import load_file

def quantize(src_dir, out_path):
    sd = load_file(os.path.join(src_dir, "model.safetensors"))
    store = {}
    for k, v in sd.items():
        w = v.float()
        if w.ndim == 2:
            s = torch.clamp(w.abs().amax(dim=1, keepdim=True) / 127.0, min=1e-8)
            store[k + "::q"] = torch.round(w / s).clamp(-127, 127).to(torch.int8).numpy()
            store[k + "::s"] = s.half().numpy()
        else:
            store[k + "::f"] = w.half().numpy()
    np.savez(out_path, **store)
    print(f"{src_dir} -> {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB)")

if __name__ == "__main__":
    quantize(sys.argv[1], sys.argv[2])
