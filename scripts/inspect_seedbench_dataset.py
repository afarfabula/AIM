import os


def main():
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    repo = os.environ.get("SEED_REPO", "lmms-lab/SEED-Bench")
    print("repo=", repo)

    try:
        cfgs = get_dataset_config_names(repo)
        print("configs=", cfgs)
    except Exception as e:
        print("get_dataset_config_names failed:", repr(e))
        cfgs = [None]

    # Check splits for first config (or default)
    cfg0 = cfgs[0] if cfgs else None
    try:
        splits = get_dataset_split_names(repo, cfg0) if cfg0 else get_dataset_split_names(repo)
        print("splits=", splits)
    except Exception as e:
        print("get_dataset_split_names failed:", repr(e))
        splits = ["test"]

    # Load a tiny sample and inspect data_type values
    split0 = splits[0] if splits else "test"
    print("loading split=", split0, "config=", cfg0)
    ds = load_dataset(repo, cfg0, split=split0)
    print("len=", len(ds))
    n = min(50, len(ds))
    types = {}
    for i in range(n):
        dt = ds[i].get("data_type", None)
        types[str(dt)] = types.get(str(dt), 0) + 1
    print("data_type_counts_first50=", types)


if __name__ == "__main__":
    main()

