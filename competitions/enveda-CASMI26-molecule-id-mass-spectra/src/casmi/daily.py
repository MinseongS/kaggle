"""Mechanical part of the daily automation (see DAILY.md).

  uv run python -m casmi.daily status            # scores -> EXPERIMENTS.md, LB snapshot, auto-submit queued kernels
  uv run python -m casmi.daily status --dry-run  # same, but never submits
  uv run python -m casmi.daily enqueue --kernel mins00/casmi26-x --version 3 --exp exp015 --msg "exp015: ..."
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

COMP = "enveda-CASMI26-molecule-id-mass-spectra"
ROOT = Path(__file__).resolve().parents[2]
AUTO = ROOT / "automation"
QUEUE = AUTO / "queue.json"
LB_HIST = AUTO / "lb_history.csv"
EXPS = ROOT / "EXPERIMENTS.md"
DAILY_LIMIT = 5
ME = "mins00"


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    out = "\n".join(l for l in r.stdout.splitlines() if not l.startswith("Warning: Looks like"))
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {r.stderr.strip()[:500]}")
    return out


def submissions() -> list[dict]:
    return list(csv.DictReader(io.StringIO(kaggle("competitions", "submissions", COMP, "-v"))))


def used_today(subs: list[dict]) -> int:
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(s["date"].startswith(today) and "ERROR" not in s["status"] for s in subs)


def update_experiments(subs: list[dict]) -> list[str]:
    """Fill `대기` LB cells of EXPERIMENTS.md rows whose exp id matches a scored submission."""
    scored = {}
    for s in subs:
        m = re.match(r"(exp\d+[\w-]*)", s["description"] or "")
        if m and s["publicScore"]:
            scored.setdefault(m.group(1), s["publicScore"])
    lines, changed = EXPS.read_text().splitlines(), []
    for i, line in enumerate(lines):
        cells = line.split("|")
        if len(cells) < 7 or "대기" not in cells[5]:
            continue
        exp = cells[2].strip()
        key = next((k for k in scored if exp == k or exp.startswith(k + "-") or k.startswith(exp)), None)
        if key:
            cells[5] = f" **{scored[key]}** "
            lines[i] = "|".join(cells)
            changed.append(f"{exp} -> {scored[key]}")
    if changed:
        EXPS.write_text("\n".join(lines) + "\n")
    return changed


def leaderboard_snapshot() -> dict:
    with tempfile.TemporaryDirectory() as d:
        kaggle("competitions", "leaderboard", COMP, "-d", "-p", d, "-q")
        z = next(Path(d).glob("*.zip"))
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            rows = list(csv.DictReader(io.TextIOWrapper(zf.open(name), encoding="utf-8-sig")))
    scores = sorted((float(r["Score"]) for r in rows), reverse=True)
    n = len(scores)
    gold_rank = 10 + int(n * 0.002)
    silver_rank, bronze_rank = int(n * 0.05), int(n * 0.10)
    mine = next((r for r in rows if ME in (r.get("TeamMemberUserNames") or "") or r.get("TeamName") == ME), None)
    snap = dict(
        utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), teams=n, top=scores[0],
        gold=scores[gold_rank - 1], silver=scores[silver_rank - 1], bronze=scores[bronze_rank - 1],
        my_rank=int(mine["Rank"]) if mine and mine.get("Rank") else "", my_score=mine["Score"] if mine else "",
        silver_rank=silver_rank,
    )
    new = not LB_HIST.exists()
    with LB_HIST.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(snap))
        if new:
            w.writeheader()
        w.writerow(snap)
    return snap


def load_queue() -> list[dict]:
    return json.loads(QUEUE.read_text()) if QUEUE.exists() else []


def save_queue(q: list[dict]) -> None:
    QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n")


def kernel_status(ref: str) -> str:
    out = kaggle("kernels", "status", ref, check=False)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', out)
    return m.group(1) if m else out.strip()[:120]


def process_queue(subs: list[dict], dry: bool) -> list[str]:
    q, log = load_queue(), []
    left = DAILY_LIMIT - used_today(subs)
    for item in q:
        if item.get("state") != "queued":
            continue
        st = kernel_status(f"{item['kernel']}/{item['version']}")
        item["last_status"] = st
        if st in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED"):
            item["state"] = "failed"
            log.append(f"FAILED {item['exp']} ({item['kernel']} v{item['version']}): {st}")
        elif st == "COMPLETE":
            if left <= 0:
                log.append(f"READY {item['exp']} but daily limit reached")
            elif dry:
                log.append(f"DRY-RUN would submit {item['exp']}")
            else:
                kaggle("competitions", "submit", COMP, "-k", item["kernel"], "-v", str(item["version"]),
                       "-f", item.get("file", "submission.csv"), "-m", item["msg"])
                item["state"] = "submitted"
                item["submitted_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
                left -= 1
                log.append(f"SUBMITTED {item['exp']} ({item['kernel']} v{item['version']})")
        else:
            log.append(f"WAIT {item['exp']}: {st}")
    save_queue(q)
    return log


def cmd_status(dry: bool) -> None:
    AUTO.mkdir(exist_ok=True)
    subs = submissions()
    print("== new scores in EXPERIMENTS.md:", update_experiments(subs) or "none")
    print("== queue:")
    for l in process_queue(subs, dry) or ["(empty)"]:
        print("  ", l)
    subs = submissions()
    print(f"== submissions used today (UTC): {used_today(subs)}/{DAILY_LIMIT}")
    print("== latest submissions:")
    for s in subs[:6]:
        print(f"   {s['date'][:16]}  {s['status'].split('.')[-1]:<9} {s['publicScore'] or '-':>6}  {s['description'][:90]}")
    best = max((float(s["publicScore"]) for s in subs if s["publicScore"]), default=None)
    snap = leaderboard_snapshot()
    print(f"== leaderboard {snap['utc']} UTC: teams {snap['teams']}, top {snap['top']}, gold {snap['gold']}, "
          f"silver {snap['silver']} (rank {snap['silver_rank']}), bronze {snap['bronze']}")
    print(f"== me: rank {snap['my_rank']} score {snap['my_score']} (best submission public {best})")


def cmd_enqueue(a: argparse.Namespace) -> None:
    AUTO.mkdir(exist_ok=True)
    q = load_queue()
    q.append(dict(exp=a.exp, kernel=a.kernel, version=a.version, msg=a.msg, file=a.file, state="queued",
                  queued_utc=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")))
    save_queue(q)
    print("queued", a.exp, a.kernel, "v", a.version)


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("status")
    s.add_argument("--dry-run", action="store_true")
    e = sp.add_parser("enqueue")
    e.add_argument("--kernel", required=True)
    e.add_argument("--version", type=int, required=True)
    e.add_argument("--exp", required=True)
    e.add_argument("--msg", required=True)
    e.add_argument("--file", default="submission.csv")
    a = p.parse_args()
    if a.cmd == "status":
        cmd_status(a.dry_run)
    else:
        cmd_enqueue(a)


if __name__ == "__main__":
    main()
