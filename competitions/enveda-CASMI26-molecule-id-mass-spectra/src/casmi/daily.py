"""Mechanical part of the daily automation (see DAILY.md).

  uv run python -m casmi.daily health            # is the improvement loop turning? (DAILY.md step 5)
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
from datetime import datetime, timedelta, timezone
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


KST = timezone(timedelta(hours=9))  # run timestamps in outputs/daily are local time
MECH_START = datetime(2026, 11, 5, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 12, 14, 23, 59, tzinfo=timezone.utc)
CHECKPOINT = datetime(2026, 11, 15, tzinfo=timezone.utc)
GPU_WEEKLY = 15.0


def _flag(level: str, name: str, msg: str) -> tuple[str, str, str]:
    return ({"red": "🟥", "yellow": "🟨", "green": "🟩"}[level], name, msg)



ORCA_AUTOMATION = "ecee5fe1-8447-437e-9dcd-077d584b1159"


def _orca_runs(hours: float = 26.0) -> dict:
    """Status counts of this competition's Orca automation runs scheduled in the last `hours` (empty on any error)."""
    import json as _json
    import subprocess
    import time as _time
    try:
        r = subprocess.run(["orca", "automations", "runs", "--id", ORCA_AUTOMATION, "--json"],
                           capture_output=True, text=True, timeout=30)
        runs = _json.loads(r.stdout)["result"]["runs"]
    except Exception as e:  # noqa: BLE001 - health must never crash
        return {"error": type(e).__name__}
    cut = (_time.time() - hours * 3600) * 1000
    out: dict = {}
    for x in runs:
        if (x.get("scheduledFor") or 0) >= cut:
            st = str(x.get("status"))
            key = "failed" if st in ("failed", "error", "skipped", "missed", "cancelled") else st
            out[key] = out.get(key, 0) + 1
    return out

def cmd_health() -> None:
    """Is the loop (directives -> experiments -> submissions -> pipeline gains) actually turning?"""
    now = datetime.now(timezone.utc)
    out: list[tuple[str, str, str]] = []
    hrs = lambda t: (now - t).total_seconds() / 3600

    # 0. directives inbox
    d = (ROOT / "DIRECTIVES.md").read_text()
    inbox = d.split("## 받은 편지함")[1].split("## 반영됨")[0]
    pending = [l for l in inbox.splitlines() if re.match(r"\s*- \d{4}-\d{2}-\d{2}", l)]
    out.append(_flag("red", "지시", f"미반영 지시 {len(pending)}건") if pending else _flag("green", "지시", "받은 편지함 비어 있음"))

    # 1. submissions
    subs = submissions()
    dates = [datetime.strptime(s["date"][:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) for s in subs]
    last = max(dates) if dates else None
    h = hrs(last) if last else 1e9
    n48 = sum(hrs(t) <= 48 for t in dates)
    lvl = "red" if h > 72 else "yellow" if h > 48 else "green"
    out.append(_flag(lvl, "제출", f"마지막 제출 {h:.0f}시간 전, 최근 48시간 {n48}회 (한도 {DAILY_LIMIT}/일)"))

    # 2. queue health
    q = load_queue()
    stuck = [i for i in q if i.get("state") == "queued"
             and hrs(datetime.strptime(i["queued_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)) > 24]
    failed7 = [i for i in q if i.get("state") == "failed"
               and hrs(datetime.strptime(i["queued_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)) < 168]
    lvl = "red" if stuck or len(failed7) >= 2 else "yellow" if failed7 else "green"
    out.append(_flag(lvl, "대기열", f"24시간 넘게 대기 {[i['exp'] for i in stuck]}, 최근 7일 실패 {[i['exp'] for i in failed7]}"))

    # 3. experiment activity (EXPERIMENTS.md rows by date)
    rows = [l for l in EXPS.read_text().splitlines() if re.match(r"\| \d{4}-\d{2}-\d{2} \|", l)]
    rdates = [datetime.strptime(l[2:12], "%Y-%m-%d").replace(tzinfo=timezone.utc) for l in rows]
    n2 = sum(hrs(t) <= 48 + 24 for t in rdates)  # date granularity: allow one extra day
    out.append(_flag("red" if n2 == 0 else "green", "실험", f"최근 2일 기록 {n2}행 (전체 {len(rows)}행)"))

    # 4. plan freshness
    qa = hrs(datetime.fromtimestamp((ROOT / "QUEUE.md").stat().st_mtime, timezone.utc))
    out.append(_flag("yellow" if qa > 72 else "green", "계획", f"QUEUE.md 마지막 수정 {qa:.0f}시간 전"))

    # 5. automation runs actually happening
    dd = ROOT / "outputs/daily"
    reps = sorted(dd.glob("report-*.md")) if dd.exists() else []
    recent = [p for p in reps if hrs(datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)) <= 26]
    logs = sorted(dd.glob("run-*.log")) if dd.exists() else []
    bad = [p.name for p in logs[-2:] if p.stat().st_size == 0 or re.search(r"Traceback|rate.?limit|Error:|invalid api key|not logged in|authentication_error", p.read_text(errors="ignore")[-4000:], re.I)]
    fails = (dd / "failures.log").read_text().splitlines()[-3:] if (dd / "failures.log").exists() else []
    started = hrs(MECH_START) > 26
    lvl = "red" if started and not recent else "yellow" if bad else "green"
    lvl = "red" if any(hrs(datetime.strptime(f[:13], "%Y%m%d-%H%M").replace(tzinfo=KST)) < 24 for f in fails) else lvl
    orca = _orca_runs()
    if orca.get("failed"):
        lvl = "red"
    out.append(_flag(lvl, "자동 실행", f"최근 26시간 리포트 {len(recent)}개 (기대 2), 문제 로그 {bad}, 최근 실패 {fails}, Orca {orca}"))

    # 6. outcome: score / rank trend
    if LB_HIST.exists():
        hist = list(csv.DictReader(LB_HIST.open()))
        cur = hist[-1]
        week = [r for r in hist if hrs(datetime.strptime(r["utc"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)) >= 168]
        prev = week[-1] if week else hist[0]
        gap = float(cur["my_score"] or 0) - float(cur["silver"])
        improved = float(cur["my_score"] or 0) > float(prev["my_score"] or 0)
        lvl = "green" if gap > 0.01 else "yellow" if improved or hrs(MECH_START) < 168 else "red"
        out.append(_flag(lvl, "성과", f"public {cur['my_score']} 순위 {cur['my_rank']} / 은 경계 {cur['silver']} (차이 {gap:+.3f}), "
                                      f"비교 시점({prev['utc']}) {prev['my_score']} 순위 {prev['my_rank']}"))
    recent_lb = [(s["description"][:40], s["publicScore"]) for s, t in zip(subs, dates) if hrs(t) <= 14 * 24 and s["publicScore"]]
    out.append(_flag("green", "최근 LB", f"{recent_lb[:8]}"))

    # 6b. CV<->LB direction: every submission is CV-pass by rule, so an LB below the best
    # earlier LB (within 14 days) is a direction mismatch. 2+ of the last 3 -> suspect the CV harness.
    scored = []
    for line in EXPS.read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 5 and re.fullmatch(r"exp\d+(-\S+)?", c[1]) and re.fullmatch(r"\*{0,2}[0-9.]+\*{0,2}", c[4].split()[0] if c[4] else ""):
            try:
                scored.append((datetime.strptime(c[0], "%Y-%m-%d"), c[1], float(c[4].split()[0].strip("*"))))
            except ValueError:
                pass
    checks = []
    for i, (d, e, lb) in enumerate(scored):
        prior = [s for s in scored[:i] if timedelta(0) <= d - s[0] <= timedelta(days=14)]
        if prior:
            best = max(prior, key=lambda s: s[2])
            checks.append((e, lb, best[1], best[2], lb >= best[2]))
    last = checks[-3:]
    miss = [c for c in last if not c[4]]
    lvl = "red" if len(miss) >= 2 else "yellow" if miss else "green"
    out.append(_flag(lvl, "CV↔LB", f"최근 {len(last)}개 중 LB 가 기준 아래 {len(miss)}개: "
                     + ", ".join(f"{e} {lb:.3f} vs {be} {bl:.3f}" for e, lb, be, bl, _ in last)
                     + (" → CV 하네스 점검을 QUEUE 맨 위로" if lvl == "red" else "")))

    # 7. GPU budget
    g = AUTO / "gpu_log.csv"
    used = 0.0
    if g.exists():
        for r in csv.DictReader(g.open()):
            try:
                if hrs(datetime.fromisoformat(r["date"]).replace(tzinfo=timezone.utc)) <= 168:
                    used += float(r["est_hours"])
            except (KeyError, ValueError):
                pass
    out.append(_flag("red" if used > GPU_WEEKLY else "yellow" if used > 0.8 * GPU_WEEKLY else "green",
                     "GPU", f"최근 7일 {used:.1f} / {GPU_WEEKLY:.0f}시간"))

    # 8. retro + uncommitted + calendar
    retros = sorted(dd.glob("retro-*.md")) if dd.exists() else []
    last_retro = datetime.strptime(retros[-1].stem[6:14], "%Y%m%d").replace(tzinfo=timezone.utc) if retros else MECH_START
    due = hrs(last_retro) >= 168
    out.append(_flag("yellow" if due else "green", "회고", ("주간 회고 필요" if due else "회고 불필요") + f" (마지막 {last_retro:%m-%d})"))
    st = subprocess.run(["git", "status", "--short", "--", "."], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    out.append(_flag("yellow" if st else "green", "미커밋", f"{len(st)}개 파일 (커밋은 사용자 요청 시에만)"))
    out.append(_flag("yellow" if 0 < (CHECKPOINT - now).days <= 3 else "green", "일정",
                     f"중간 점검(11/15)까지 {(CHECKPOINT - now).days}일, 마감까지 {(DEADLINE - now).days}일"))

    for icon, name, msg in out:
        print(f"{icon} {name:<6} {msg}")
    n_red = sum(i == "🟥" for i, _, _ in out)
    print(f"== health: 🟥 {n_red}, 🟨 {sum(i == '🟨' for i, _, _ in out)}")


def main() -> None:
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("health")
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
    elif a.cmd == "health":
        cmd_health()
    else:
        cmd_enqueue(a)


if __name__ == "__main__":
    main()
