#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


DEFAULT_STRATEGIES = [
    f"bishemethod_v2stage_anchor16_aware_v{i}" for i in range(1, 11)
]


def parse_args():
    parser = argparse.ArgumentParser(description="Queue full MME runs for anchor-aware variants.")
    parser.add_argument("--worker-id", required=True, help="Target mlx worker id.")
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids on the worker.")
    parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated strategy names to run.",
    )
    parser.add_argument("--tasks", default="", help="Comma-separated task list passed via --tasks.")
    parser.add_argument("--bench", default="mme", help="Benchmark name passed to run_llava_next.sh.")
    parser.add_argument("--limit", default="none", help="Limit passed to run_llava_next.sh.")
    parser.add_argument("--attn", default="sdpa", help="Attention implementation.")
    parser.add_argument("--model", default="liuhaotian/llava-v1.6-vicuna-13b", help="Model repo/path passed to run_llava_next.sh.")
    parser.add_argument("--model-name", default="llava-v1.6-vicuna-13b", help="LLaVA builder model name.")
    parser.add_argument(
        "--root",
        default="/mlx_devbox/users/quyanyi/playground/AIM",
        help="Project root on both local machine and worker.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval in seconds.",
    )
    return parser.parse_args()


def build_remote_command(root: str, gpu: int, bench: str, tasks: str, limit: str, strategy: str, attn: str, model: str, model_name: str) -> str:
    root_q = shlex.quote(root)
    strategy_q = shlex.quote(strategy)
    bench_q = shlex.quote(bench)
    tasks_q = shlex.quote(tasks)
    limit_q = shlex.quote(limit)
    attn_q = shlex.quote(attn)
    model_q = shlex.quote(model)
    model_name_q = shlex.quote(model_name)
    task_part = f"--tasks {tasks_q}" if tasks else f"--bench {bench_q}"
    return (
        f"cd {root_q} && "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"bash executable/run_llava_next.sh "
        f"--model {model_q} --model-name {model_name_q} "
        f"{task_part} --limit {limit_q} --strategy {strategy_q} --attn {attn_q}"
    )


def launch_one(worker_id: str, root: str, gpu: int, bench: str, tasks: str, limit: str, strategy: str, attn: str, model: str, model_name: str, queue_log_dir: Path):
    remote_cmd = build_remote_command(root, gpu, bench, tasks, limit, strategy, attn, model, model_name)
    cmd = ["mlx", "worker", "login", worker_id, "--", remote_cmd]
    log_path = queue_log_dir / f"{strategy}_gpu{gpu}.log"
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f, log_path


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_log_dir = Path(root) / "logs" / f"anchor_mme_queue_{args.worker_id}_{ts}"
    queue_log_dir.mkdir(parents=True, exist_ok=True)

    pending = deque(strategies)
    active = {}

    print(f"[queue] worker={args.worker_id} gpus={gpus} strategies={len(strategies)}")
    print(f"[queue] local logs: {queue_log_dir}")
    sys.stdout.flush()

    def maybe_launch(gpu_id: int):
        if not pending:
            return
        strategy = pending.popleft()
        proc, log_f, log_path = launch_one(
            worker_id=args.worker_id,
            root=root,
            gpu=gpu_id,
            bench=args.bench,
            tasks=args.tasks,
            limit=args.limit,
            strategy=strategy,
            attn=args.attn,
            model=args.model,
            model_name=args.model_name,
            queue_log_dir=queue_log_dir,
        )
        active[gpu_id] = {
            "strategy": strategy,
            "proc": proc,
            "log_f": log_f,
            "log_path": log_path,
            "start_time": time.time(),
        }
        print(f"[launch] gpu={gpu_id} strategy={strategy} pid={proc.pid} log={log_path}")
        sys.stdout.flush()

    for gpu in gpus:
        maybe_launch(gpu)

    try:
        while active:
            time.sleep(args.poll_seconds)
            for gpu in list(active.keys()):
                info = active[gpu]
                ret = info["proc"].poll()
                if ret is None:
                    elapsed = int(time.time() - info["start_time"])
                    print(f"[running] gpu={gpu} strategy={info['strategy']} elapsed={elapsed}s")
                    continue

                info["log_f"].close()
                elapsed = int(time.time() - info["start_time"])
                print(
                    f"[done] gpu={gpu} strategy={info['strategy']} exit={ret} "
                    f"elapsed={elapsed}s log={info['log_path']}"
                )
                del active[gpu]
                maybe_launch(gpu)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("[queue] interrupted, terminating active child processes")
        for info in active.values():
            info["proc"].terminate()
            info["log_f"].close()
        raise

    print("[queue] all strategies completed")
    print(f"[queue] local logs: {queue_log_dir}")


if __name__ == "__main__":
    main()
