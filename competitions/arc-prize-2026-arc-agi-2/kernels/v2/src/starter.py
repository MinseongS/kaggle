import bz2
import glob
import os
import pickle
import time
import json
import torch
import argparse
import torch.multiprocessing as mp
from collections import Counter, defaultdict

DEBUG_KEYS = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]
DIR_OUTPUTS = "/kaggle/inference_outputs"
# Start another pass only if at least this many seconds remain.
MIN_PASS_REMAINING = float(os.getenv("ARC_MIN_PASS_REMAINING", "1800"))
MAX_PASSES = int(os.getenv("ARC_MAX_PASSES", "2"))


def local_worker(rank, queue, end_time):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)

    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")

    worker(rank, queue, end_time)

    print(f"[Rank {rank}] done!")


def estimated_work(task):
    # Rough proxy for TTT + decode cost: grid cells seen in training (x16 augments)
    # plus cells decoded per test output.
    train_cells = sum(len(p["input"]) * len(p["input"][0]) + len(p["output"]) * len(p["output"][0]) for p in task["train"])
    test_cells = sum(len(t["input"]) * len(t["input"][0]) for t in task["test"])
    return train_cells * 16 + test_cells * 8 * len(task["test"])


def vote_margin(n_outputs_by_key):
    """Per task: smallest (top-1 votes - top-2 votes) over its test outputs; -1 if an output has no candidate.

    A vote is one augmented view whose DFS produced that grid, as counted by score_kgmon.
    """
    votes = defaultdict(Counter)
    for path in glob.glob(os.path.join(DIR_OUTPUTS, "*")):
        with bz2.BZ2File(path) as f:
            for g in pickle.load(f):
                votes[os.path.basename(path).split(".")[0]][tuple(map(tuple, g["solution"]))] += 1
    margins = {}
    for key, n_outputs in n_outputs_by_key.items():
        per_output = []
        for i in range(n_outputs):
            top = [c for _, c in votes[f"{key}_{i}"].most_common(2)] + [0, 0]
            per_output.append(top[0] - top[1] if top[0] else -1)
        margins[key] = min(per_output)
    return margins


def run_pass(n_pass, keys, nprocs, end_time):
    os.environ["ARC_PASS"] = str(n_pass)
    # Start-up markers left by the previous pass would let every rank patch Unsloth at once.
    for marker in glob.glob("/kaggle/worker*"):
        os.remove(marker)
    print(f"*** pass {n_pass}: {len(keys)} tasks on {nprocs} GPUs, {end_time - time.time():.0f}s left")
    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(nprocs):
        queue.put(None)
    mp.spawn(local_worker, args=(queue, end_time), nprocs=nprocs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    keys = list(data.keys())
    if not rerun_mode:
        # ARC_EVAL_KEYS: "debug" (4 tasks, default), "all", or comma-separated task ids.
        eval_keys = os.getenv("ARC_EVAL_KEYS", "debug")
        if eval_keys == "debug":
            keys = DEBUG_KEYS
        elif eval_keys != "all":
            keys = eval_keys.split(",")

    # Cheap tasks first: when time runs out, the tasks left undone are the expensive ones
    # instead of whatever sorts last alphabetically.
    keys = sorted(keys, key=lambda k: (estimated_work(data[k]), k))

    nprocs = torch.cuda.device_count() or 4

    run_pass(1, keys, nprocs, args.end_time)

    # Spend leftover time re-solving the least certain tasks first, with new seeds.
    for n_pass in range(2, MAX_PASSES + 1):
        if args.end_time - time.time() < MIN_PASS_REMAINING:
            break
        margins = vote_margin({k: len(data[k]["test"]) for k in keys})
        print(f"*** vote margins before pass {n_pass}: {sorted(Counter(margins.values()).items())}")
        order = sorted(keys, key=lambda k: (margins[k], estimated_work(data[k]), k))
        run_pass(n_pass, order, nprocs, args.end_time)
