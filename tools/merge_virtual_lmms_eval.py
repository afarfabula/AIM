#!/home/tiger/miniforge3/envs/aim/bin/python
import argparse
import collections
import json
from pathlib import Path

from loguru import logger as eval_logger

from lmms_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
)
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.utils import get_datetime_str, get_git_commit_hash, make_table


def parse_args():
    parser = argparse.ArgumentParser(description="Merge single-GPU virtual data-parallel lmms-eval shards.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing shard_* subdirectories.")
    parser.add_argument("--model", default="llava", help="Model name used by lmms-eval.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task list. Defaults to the tasks from the first shard result.")
    parser.add_argument("--output_path", default=None, help="Directory to save merged results. Defaults to <run_dir>/merged.")
    parser.add_argument("--include_path", default=None, help="Optional extra task path passed to TaskManager.")
    parser.add_argument("--verbosity", default="INFO", help="Logger verbosity.")
    parser.add_argument("--bootstrap_iters", type=int, default=100000, help="Bootstrap iterations for stderr.")
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_files(run_dir: Path):
    shard_state_files = sorted(run_dir.glob("shard_*/**/*_virtual_shard_rank*.json"))
    if not shard_state_files:
        raise FileNotFoundError(f"No virtual shard state files found under {run_dir}")

    result_files = sorted(run_dir.glob("shard_*/**/*_results.json"))
    if not result_files:
        raise FileNotFoundError(f"No shard result files found under {run_dir}")
    return shard_state_files, result_files


def _merge_task_outputs(task_dict, shard_states, bootstrap_iters):
    eval_tasks = get_task_list(task_dict)
    for task_output in eval_tasks:
        merged_metrics = collections.defaultdict(list)
        merged_logged_samples = []
        for shard_state in shard_states:
            task_state = shard_state["task_state"].get(task_output.task_name)
            if task_state is None:
                continue
            for metric_entry in task_state.get("sample_metrics", []):
                key = (metric_entry["metric"], metric_entry["filter_key"])
                merged_metrics[key].extend(metric_entry["values"])
            merged_logged_samples.extend(task_state.get("logged_samples", []))
        if not merged_metrics:
            raise ValueError(f"No shard metrics found for task {task_output.task_name}")
        task_output.sample_metrics = merged_metrics
        task_output.logged_samples = merged_logged_samples
        task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
    return eval_tasks


def main():
    args = parse_args()
    eval_logger.remove()
    eval_logger.add(lambda msg: print(msg, end=""), level=args.verbosity)

    run_dir = Path(args.run_dir).resolve()
    shard_state_files, result_files = _discover_files(run_dir)
    shard_states = [_load_json(path) for path in shard_state_files]
    first_results = _load_json(result_files[0])

    world_sizes = {state["world_size"] for state in shard_states}
    if len(world_sizes) != 1:
        raise ValueError(f"Inconsistent virtual world sizes: {sorted(world_sizes)}")
    world_size = world_sizes.pop()
    ranks = sorted(state["rank"] for state in shard_states)
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"Duplicate virtual ranks detected: {ranks}")
    if len(shard_states) != world_size:
        eval_logger.warning(f"Expected {world_size} shards, found {len(shard_states)}. Proceeding with available shards.")

    if args.tasks:
        task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    else:
        task_names = list(first_results["configs"].keys())

    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)
    task_dict = get_task_dict(task_names, task_manager)
    eval_tasks = _merge_task_outputs(task_dict, shard_states, bootstrap_iters=args.bootstrap_iters)

    results, samples, configs, versions, num_fewshot, higher_is_better = consolidate_results(eval_tasks)
    if bool(results):
        results, versions, show_group_table, *_ = consolidate_group_results(results, versions, task_dict)
    else:
        show_group_table = False

    results_agg, group_agg = prepare_print_tasks(task_dict, results)
    subtask_list = get_subtask_list(task_dict)

    merged_results = {
        "results": dict(results_agg.items()),
        **({"groups": dict(group_agg.items())} if (bool(group_agg) and show_group_table) else {}),
        "group_subtasks": dict(reversed(subtask_list.items())),
        "configs": dict(sorted(configs.items())),
        "versions": dict(sorted(versions.items())),
        "n-shot": dict(sorted(num_fewshot.items())),
        "higher_is_better": dict(sorted(higher_is_better.items())),
        "n-samples": {
            task_output.task_name: {
                "original": len(task_output.task.eval_docs),
                "effective": task_output.sample_len,
            }
            for task_output in eval_tasks
        },
        "config": dict(first_results.get("config", {})),
        "git_hash": get_git_commit_hash(),
        "date": get_datetime_str(),
        "model_source": first_results.get("model_source", args.model),
        "model_name": first_results.get("model_name"),
        "model_name_sanitized": first_results.get("model_name_sanitized"),
        "merged_from_virtual_shards": {
            "run_dir": str(run_dir),
            "world_size": world_size,
            "ranks": ranks,
        },
    }
    merged_results["config"]["virtual_rank"] = "merged"
    merged_results["config"]["virtual_world_size"] = world_size

    output_root = Path(args.output_path).resolve() if args.output_path else run_dir / "merged"
    model_name_sanitized = merged_results.get("model_name_sanitized") or first_results["model_name_sanitized"]
    target_dir = output_root / model_name_sanitized
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"{merged_results['date']}_results.json"
    out_file.write_text(json.dumps(merged_results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(make_table(merged_results))
    if "groups" in merged_results:
        print(make_table(merged_results, "groups"))
    print(f"Merged results saved to {out_file}")


if __name__ == "__main__":
    main()
