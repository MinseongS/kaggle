#!/usr/bin/env python3
"""CPU program-search solver driver (icecuber ARC 2020 DSL search, MIT).

Runs the icecuber C++ binary on every (task, test_index) of a Kaggle-format challenges JSON,
in several passes (depth / flip variants), with per-process time + RSS limits, a global deadline,
and a nice level so it can run next to the GPU pipeline. Stdlib only (no psutil).

Output JSON (--out):
  {task_id: [ {"cands": [{"grid": [[..]], "score": float, "fits_all": bool, "pass": "3"}...]},  # test 0
              ... ]}                                                                            # test 1..
  cands are sorted by score desc, de-duplicated, at most 3 per test.
  fits_all = candidate reproduces ALL train outputs exactly (icecuber score = #train_ok - 0.01*complexity).
Also writes <out>.stats.jsonl with one line per process run (time, peak RSS, status).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(HERE, "icecuber", "run")


def rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        return 0.0
    except FileNotFoundError:
        pass
    try:  # macOS fallback
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return int(out) / 1024 if out else 0.0
    except Exception:
        return 0.0


def parse_answer_file(fn):
    with open(fn) as f:
        lines = f.read().strip().split("\n")
    cands = []
    for line in lines[1:]:
        img, score = line.split()
        rows = [r for r in img.split("|") if r]
        grid = [[int(ch) for ch in r] for r in rows]
        cands.append((float(score), grid))
    return lines[0], cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--bin", default=DEFAULT_BIN, type=os.path.abspath)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    # pass spec: "arg:timeout_s:workers" ; arg = depth (2,3,4) or flip variants (23,33) as in icecuber safe_run
    ap.add_argument("--passes", default="3:300:0,23:300:0,33:300:0")
    ap.add_argument("--mem-mb", type=float, default=8000, help="per-process RSS limit")
    ap.add_argument("--total-mem-mb", type=float, default=48000, help="kill largest proc if sum RSS exceeds")
    ap.add_argument("--deadline", type=float, default=0, help="unix time; stop launching/kill after this")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--keys", default=None, help="comma-separated task ids subset")
    args = ap.parse_args()

    ch = json.load(open(args.challenges))
    keys = sorted(ch)
    if args.keys:
        sel = set(args.keys.split(","))
        keys = [k for k in keys if k in sel]
    work = os.path.abspath(args.workdir or (os.path.splitext(args.out)[0] + "_work"))
    tdir = os.path.join(work, "tasks")
    odir = os.path.join(work, "output")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(odir, exist_ok=True)
    for f in os.listdir(tdir):
        os.remove(os.path.join(tdir, f))
    # one file per task; icecuber splits multi-test tasks into consecutive samples (sorted by filename)
    index = []  # sample idx -> (task_id, test_idx)
    for k in keys:
        t = ch[k]
        with open(os.path.join(tdir, k + ".json"), "w") as f:
            json.dump({"train": t["train"], "test": [{"input": x["input"]} for x in t["test"]]}, f,
                      separators=(",", ":"))
        for j in range(len(t["test"])):
            index.append((k, j))
    ntrain = {k: len(ch[k]["train"]) for k in keys}

    env = dict(os.environ, ICECUBER_DATA=tdir, ICECUBER_OUT=odir, OMP_NUM_THREADS="1")
    stats_f = open(args.out + ".stats.jsonl", "a")
    lock = threading.Lock()
    running = {}  # pid -> Popen
    stop = threading.Event()

    def watchdog():
        while not stop.is_set():
            with lock:
                procs = list(running.values())
            mems = [(rss_mb(p.pid), p) for p in procs]
            if args.total_mem_mb and sum(m for m, _ in mems) > args.total_mem_mb and mems:
                m, p = max(mems, key=lambda x: x[0])
                p._killed_reason = "TOTAL_MLE"
                p.kill()
            for m, p in mems:
                if m > args.mem_mb:
                    p._killed_reason = "MLE"
                    p.kill()
                if args.deadline and time.time() > args.deadline:
                    p._killed_reason = "DEADLINE"
                    p.kill()
            time.sleep(0.5)

    def preexec():
        try:
            os.nice(args.nice)
        except Exception:
            pass
        if sys.platform.startswith("linux"):
            lim = int(args.mem_mb * 1.5 * 2**20)
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

    def run_one(i, arg, timeout):
        if args.deadline and time.time() > args.deadline - 5:
            return
        fn = os.path.join(odir, f"answer_{i}_{arg}.csv")
        if os.path.exists(fn):
            return
        t0 = time.time()
        p = subprocess.Popen([args.bin, str(i), str(arg)], env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, preexec_fn=preexec, cwd=work)
        p._killed_reason = None
        with lock:
            running[p.pid] = p
        status = "OK"
        timer = threading.Timer(timeout, lambda: (setattr(p, "_killed_reason", "TLE"), p.kill()))
        timer.start()
        _, wstatus, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(wstatus)
        timer.cancel()
        with lock:
            running.pop(p.pid, None)
        peak = ru.ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)
        if p._killed_reason:
            status = p._killed_reason
        elif p.returncode != 0:
            status = f"RTE{p.returncode}"
        if status != "OK" and os.path.exists(fn):
            os.remove(fn)
        k, j = index[i]
        rec = dict(task=k, test=j, idx=i, arg=arg, status=status, sec=round(time.time() - t0, 2),
                   peak_mb=round(peak, 1), cpu_s=round(ru.ru_utime + ru.ru_stime, 2))
        with lock:
            stats_f.write(json.dumps(rec) + "\n")
            stats_f.flush()

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()
    for spec in [s for s in args.passes.split(",") if s]:
        arg, timeout, w = spec.split(":")
        w = int(w) or args.workers
        if args.deadline and time.time() > args.deadline - 5:
            break
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=w) as ex:
            list(ex.map(lambda i: run_one(i, int(arg), float(timeout)), range(len(index))))
        print(f"pass {arg} done in {time.time()-t0:.0f}s", flush=True)
    stop.set()

    # merge
    res = {k: [{"cands": []} for _ in ch[k]["test"]] for k in keys}
    for fn in os.listdir(odir):
        if not fn.startswith("answer_"):
            continue
        _, i, arg = os.path.splitext(fn)[0].split("_")
        i = int(i)
        if i >= len(index):
            continue
        k, j = index[i]
        sid, cands = parse_answer_file(os.path.join(odir, fn))
        assert sid == f"{k}_{j}", (sid, k, j)
        for score, grid in cands:
            if score < 0:
                continue  # dummy
            res[k][j]["cands"].append(dict(grid=grid, score=score, fits_all=score > ntrain[k] - 0.5, **{"pass": arg}))
    for k in res:
        for t in res[k]:
            best = {}
            for c in sorted(t["cands"], key=lambda c: -c["score"]):
                key = json.dumps(c["grid"])
                if key not in best:
                    best[key] = dict(c, passes=[c["pass"]])
                elif c["pass"] not in best[key]["passes"]:
                    best[key]["passes"].append(c["pass"])
                    best[key]["fits_all"] = best[key]["fits_all"] or c["fits_all"]
            for c in best.values():
                c["n_passes"] = len(c["passes"])  # >=2 means also found by a flip pass (agreement signal)
            t["cands"] = list(best.values())[:3]
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"wrote {args.out}: {len(res)} tasks")


if __name__ == "__main__":
    main()
