#!/usr/bin/env python3
import argparse
import os

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--allow-pattern", action="append", default=[])
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    print(f"repo_id={args.repo_id}")
    print(f"local_dir={args.local_dir}")
    print(f"allow_patterns={args.allow_pattern or None}")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=1,
        allow_patterns=args.allow_pattern or None,
    )
    print("done")


if __name__ == "__main__":
    main()
