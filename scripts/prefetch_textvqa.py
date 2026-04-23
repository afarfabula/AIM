#!/home/tiger/miniforge3/envs/aim/bin/python
"""
Prefetch lmms-lab/textvqa dataset artifacts into HF_DATASETS_CACHE.

Goal: avoid "no respond" hangs caused by stale builder locks / lazy image fetch.
This script does NOT use GPU.
"""

import os
import sys


def _get_env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def main() -> int:
    # Keep defaults aligned with our eval runners.
    cache_root = _get_env("HF_HOME", "/tmp/aim_hf_home")
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"Failed to import datasets: {e}", file=sys.stderr)
        return 2

    dataset_path = "lmms-lab/textvqa"
    split = os.environ.get("TEXTVQA_SPLIT", "validation")
    limit = int(os.environ.get("TEXTVQA_PREFETCH_N", "50"))

    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")
    print(f"Loading {dataset_path} split={split} ...")
    ds = load_dataset(dataset_path, split=split)
    print(ds)

    # Force materialization of Image column payloads. For datasets.Image, accessing the field
    # will download referenced files lazily.
    n = min(limit, len(ds))
    print(f"Prefetching first {n} examples (touch image payloads) ...")
    touched = 0
    for i in range(n):
        ex = ds[i]
        # Common schema in lmms-lab datasets: 'image' is a datasets.Image.
        img = ex.get("image", None)
        if img is not None:
            _ = img  # touch
            touched += 1
        if (i + 1) % 10 == 0:
            print(f"... {i+1}/{n}")

    print(f"Done. touched_images={touched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

