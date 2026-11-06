import importlib.util
import json
import os
import sys
import types

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMIT = os.path.join(ROOT, "submit")
MODEL = os.path.join(SUBMIT, "model")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def require_file(path: str) -> None:
    rel = os.path.relpath(path, ROOT)
    ok = os.path.exists(path)
    print(f"{rel}: {ok}")
    if not ok:
        raise FileNotFoundError(rel)


def load_submit_module():
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        def inference_mode():
            def deco(fn):
                return fn
            return deco

        torch_stub.inference_mode = inference_mode
        sys.modules["torch"] = torch_stub
    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")
        transformers_stub.AutoTokenizer = object
        transformers_stub.AutoModelForSequenceClassification = object
        sys.modules["transformers"] = transformers_stub

    path = os.path.join(SUBMIT, "script.py")
    spec = importlib.util.spec_from_file_location("submit_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    require_file(os.path.join(SUBMIT, "script.py"))
    require_file(os.path.join(SUBMIT, "requirements.txt"))
    require_file(os.path.join(MODEL, "config.json"))

    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for name in tokenizer_files:
        require_file(os.path.join(MODEL, name))

    bias_path = os.path.join(MODEL, "bias.npy")
    if os.path.exists(bias_path):
        bias = np.load(bias_path)
        print(f"submit/model/bias.npy: True shape={bias.shape} dtype={bias.dtype}")
        if bias.shape != (14,):
            print(f"ERROR: bias.npy shape must be (14,), got {bias.shape}", file=sys.stderr)
            return 1
    else:
        print("submit/model/bias.npy: False zero fallback will be used")

    has_weight = any(
        os.path.exists(os.path.join(MODEL, name))
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    print(f"has_weight {has_weight}")

    from src.data import serialize_example as train_serialize

    submit_script = load_submit_module()
    mismatch = 0
    test_path = os.path.join(ROOT, "data", "original_data", "test.jsonl")
    if os.path.exists(test_path):
        with open(test_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if train_serialize(sample) != submit_script.serialize_example(sample):
                    mismatch += 1
    else:
        print("data/original_data/test.jsonl: False serializer comparison skipped")

    print(f"serializer_mismatch {mismatch}")
    print(f"submit/script.py MAX_LEN {getattr(submit_script, 'MAX_LEN', None)}")
    return 1 if mismatch > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
