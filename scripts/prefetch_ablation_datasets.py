import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    name: str
    repo_id: str
    config: str | None
    split: str
    token_required: bool = False


SPECS = [
    Spec(
        name="gqa_testdev",
        repo_id="lmms-lab/GQA",
        config="testdev_balanced_instructions",
        split="testdev",
        token_required=False,
    ),
    Spec(name="mme_test", repo_id="lmms-lab/MME", config=None, split="test", token_required=True),
    Spec(name="pope_test", repo_id="lmms-lab/POPE", config=None, split="test", token_required=False),
    Spec(
        name="scienceqa_img_test",
        repo_id="lmms-lab/ScienceQA",
        config="ScienceQA-IMG",
        split="test",
        token_required=False,
    ),
    Spec(name="mmmu_val", repo_id="lmms-lab/MMMU", config=None, split="validation", token_required=False),
]


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def main():
    from datasets import load_dataset

    hf_endpoint = _env("HF_ENDPOINT", "")
    hf_home = _env("HF_HOME", "")
    ds_cache = _env("HF_DATASETS_CACHE", "")
    hub_cache = _env("HF_HUB_CACHE", "")
    print("HF_ENDPOINT=", hf_endpoint)
    print("HF_HOME=", hf_home)
    print("HF_DATASETS_CACHE=", ds_cache)
    print("HF_HUB_CACHE=", hub_cache)
    print()

    for spec in SPECS:
        print(f"== prefetch {spec.name} ==")
        if spec.token_required:
            token = _env("HF_TOKEN") or _env("HUGGINGFACE_HUB_TOKEN")
            if not token:
                raise RuntimeError(f"{spec.name} requires HF token; set HF_TOKEN/HUGGINGFACE_HUB_TOKEN")
        else:
            token = None

        # Force materialization so files are actually downloaded and cached.
        ds = load_dataset(
            spec.repo_id,
            spec.config,
            split=spec.split,
            token=token if spec.token_required else None,
        )
        _ = len(ds)
        _ = ds[0]
        print(f"download_ok: len={len(ds)} first_keys={list(ds[0].keys())[:10]}")
        print()

    print("ALL_DONE")


if __name__ == "__main__":
    main()

