import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp

DEBUG_KEYS = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]


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
    print(f"{len(keys)} tasks on {nprocs} GPUs")

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(nprocs):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=nprocs)
