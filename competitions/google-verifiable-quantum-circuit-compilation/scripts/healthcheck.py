"""자동화 상태 점검. outputs/ops/status.md 를 쓰고 추이를 outputs/ops/history.tsv 에 한 줄 쌓는다.
LB 결과가 나온 자동 제출은 EXPERIMENTS.md 의 '(대기)' 를 채운다.

Usage: uv run python scripts/healthcheck.py
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

COMP = "google-verifiable-quantum-circuit-compilation"
ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "outputs" / "ops"
H = ROOT / "outputs" / "harvest"
NOW = dt.datetime.now()


def sh(*args: str, timeout: int = 300) -> str:
    try:
        r = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception as e:
        return f"ERROR {e}"


AUTOMATIONS = {  # Orca 자동화 (orca automations list 로 확인)
    "수집·제출": ("84d89b67-4116-42c0-8e23-4d5a51ec6485", 14),
    "밤 연구": ("1440a468-bdcf-4ea4-b004-c01e8d72535f", 26),
    "아침 감독": ("9e75896d-609b-4ab7-9cb1-a8f9f69cbcac", 26),
}


def orca_json(*args: str):
    import json
    out = sh("orca", *args, "--json", timeout=60)
    try:
        return json.loads(out).get("result")
    except Exception:
        return None


def orca_status(name: str, aid: str, max_age_h: float, issues: list[str]) -> str:
    a = orca_json("automations", "show", aid)
    if not a:
        issues.append(f"Orca 자동화 '{name}' 조회 실패 (Orca 실행 중인지 확인)")
        return "조회 실패"
    a = a.get("automation", a)
    nxt = a.get("nextRunAt")
    nxt_s = dt.datetime.fromtimestamp(nxt / 1000).strftime("%m-%d %H:%M") if nxt else "?"
    if not a.get("enabled"):
        issues.append(f"Orca 자동화 '{name}' 비활성")
    runs = orca_json("automations", "runs", "--id", aid) or []
    runs = runs.get("runs", runs) if isinstance(runs, dict) else runs
    if not runs:
        return f"활성={a.get('enabled')}, 다음 {nxt_s}, 실행 기록 없음"
    last = max(runs, key=lambda r: r.get("createdAt") or 0)
    st = last.get("status")
    when = dt.datetime.fromtimestamp((last.get("createdAt") or 0) / 1000)
    age = (NOW - when).total_seconds() / 3600
    if st not in ("completed", "running", "dispatched", "pending"):
        issues.append(f"Orca '{name}' 최근 실행 상태 {st}: {(last.get('error') or '')[:120]}")
    if age > max_age_h:
        issues.append(f"Orca '{name}' 마지막 실행이 {age:.0f}시간 전")
    snap = ((last.get("outputSnapshot") or {}).get("content") or "").strip().replace("\n", " ")
    return f"활성={a.get('enabled')}, 다음 {nxt_s}, 최근 #{last.get('runNumber')} {st} ({when:%m-%d %H:%M}) — {snap[:200]}"


def age_h(p: Path) -> float | None:
    return (NOW.timestamp() - p.stat().st_mtime) / 3600 if p.exists() else None


def main() -> None:
    OPS.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    lines: list[str] = [f"# 상태 점검 {NOW.isoformat(timespec='minutes')}", ""]

    # 1. 예약 작업
    lines.append("## 예약 작업 (Orca 자동화)")
    for name, (aid, max_age) in AUTOMATIONS.items():
        lines.append(f"- {name}: {orca_status(name, aid, max_age, issues)}")

    # 2. harvest
    lines += ["", "## 수집·제출 (harvest)"]
    hl = H / "harvest.log"
    a = age_h(hl)
    if a is None:
        issues.append("harvest.log 없음")
    else:
        tail = hl.read_text().splitlines()[-200:]
        last_run = [x for x in tail if "portfolio local GM" in x or "missing tasks" in x]
        lines.append(f"- 마지막 로그 {a:.1f}시간 전")
        if a > 14:
            issues.append(f"harvest 로그가 {a:.0f}시간째 갱신 안 됨")
        for x in tail:
            if "best " in x and "ratio" in x:
                pass
        if last_run:
            lines.append(f"- 마지막 결과: {last_run[-1]}")
        # 이후 성공한 제출로 해소된 실패는 무시한다
        ok_idx = max((i for i, x in enumerate(tail) if "submit OK" in x), default=-1)
        fails = [x for x in tail[ok_idx + 1:] if "FAILED" in x or "Traceback" in x or "bad zip" in x]
        if fails:
            issues.append(f"harvest 오류 {len(fails)}건 (최근: {fails[-1][:160]})")
    lau = H / "launchd.log"
    if lau.exists() and "Traceback" in "\n".join(lau.read_text().splitlines()[-50:]):
        issues.append("launchd.log 에 Traceback (harvest 크래시)")

    # 3. 연구
    lines += ["", "## 밤 연구"]
    logs = sorted(glob.glob(str(ROOT / "outputs" / "research_logs" / "*.log")))
    if not logs:
        pass  # Orca 실행은 research_logs 를 남기지 않음 (launchd 시절 경로)
    else:
        lp = Path(logs[-1])
        txt = lp.read_text(errors="replace")
        ex = re.findall(r"exit=(\d+)", txt)
        lines.append(f"- 최근 로그 {lp.name}, exit={ex[-1] if ex else '진행중/없음'}, {len(txt)}자")
        if "Failed to authenticate" in txt or "OAuth" in txt:
            issues.append("연구 세션 인증 실패 → `claude setup-token` 토큰 저장 필요")
        if ex and ex[-1] not in ("0", "124"):
            issues.append(f"연구 세션 비정상 종료 exit={ex[-1]}")
    # Orca 상태와 무관하게 실제 활동(파일 수정 시각)으로 밤 연구를 판단한다.
    # Orca 가 "stopped watching" 으로 dispatch_failed 를 남겨도 세션은 계속 도는 경우가 있다 (11-06).
    win_end = NOW.replace(hour=7, minute=0, second=0, microsecond=0)
    if NOW < win_end:
        win_end -= dt.timedelta(days=1)
    win_start = win_end - dt.timedelta(hours=6, minutes=30)  # 00:30~07:00
    act = []
    for pat in ("src/*.py", "outputs/research/**/*", "RESEARCH.md"):
        for f in ROOT.glob(pat):
            try:
                m = dt.datetime.fromtimestamp(f.stat().st_mtime)
            except OSError:
                continue
            if win_start <= m <= win_end and f.name != "harvest.py":
                act.append(m)
    if act:
        span = (max(act) - min(act)).total_seconds() / 3600
        lines.append(f"- 실제 활동 {min(act):%m-%d %H:%M}~{max(act):%H:%M} ({span:.1f}시간, 파일 {len(act)}개 수정)")
        if span < 3:
            issues.append(f"밤 연구가 {span:.1f}시간만 활동 (5시간 중) → 일찍 멈춤, research_prompt 확인")
        issues[:] = [i if not i.startswith("Orca '밤 연구' 최근 실행 상태 dispatch_failed")
                     else i + " (단, 파일 활동은 있음 → 세션은 실제로 돌았음)" for i in issues]
    else:
        issues.append("밤 연구 시간대(00:30~07:00)에 파일 활동 없음 → 연구 세션이 돌지 않았음")
    cands = sorted((ROOT / "outputs" / "candidates").glob("*")) if (ROOT / "outputs" / "candidates").exists() else []
    lines.append(f"- 후보 폴더 {len(cands)}개: {', '.join(p.name for p in cands[-5:]) or '없음'}")
    auth = sh("bash", "-lc", "claude -p ok --max-budget-usd 0.05 < /dev/null 2>&1 | head -3", timeout=90)
    if "Failed to authenticate" in auth or "expired" in auth:
        issues.append("Claude 로그인 만료/실패 → Orca 자동화(연구·감독·수집)가 멈춤. `claude auth login` 필요")
    elif "sign-in ends" in auth:
        m = re.search(r"sign-in ends between (\S+) and (\S+)", auth)
        issues.append(f"Claude 로그인 만료 예정 {m.group(1) if m else ''}~{m.group(2) if m else ''} → 그 전에 다시 로그인 필요")

    # 4. 제출과 LB
    lines += ["", "## 제출 (최근 8개)"]
    subs_csv = sh("kaggle", "competitions", "submissions", "-c", COMP, "--csv")
    rows = list(csv.DictReader(io.StringIO("\n".join(l for l in subs_csv.splitlines() if not l.startswith("Warning")))))
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    n_today = sum(1 for r in rows if r.get("date", "").startswith(today))
    lines.append(f"- 오늘(UTC) 제출 {n_today}/5")
    best_lb = 0.0
    exp = ROOT / "EXPERIMENTS.md"
    exp_txt = exp.read_text()
    for r in rows[:8]:
        desc, score, status = r.get("description", ""), r.get("publicScore", ""), r.get("status", "")
        lines.append(f"- {r.get('date','')[:16]} | {status.split('.')[-1]} | LB {score or '-'} | {desc[:90]}")
        m = re.search(r"GM (\d+\.\d+)", desc)
        if score and m:
            gap = float(score) - float(m.group(1))
            handled = m.group(1) in (ROOT / "outputs" / "harvest" / "lb_exclude.json").read_text() \
                if (ROOT / "outputs" / "harvest" / "lb_exclude.json").exists() else False
            if abs(gap) > 0.02 and not handled:
                issues.append(f"LB-로컬 차이 {gap:+.4f} ({desc[:60]}) → salt 민감 회로 의심")
            elif abs(gap) > 0.02:
                lines.append(f"  - LB-로컬 차이 {gap:+.4f}: 해당 dqi 는 lb_exclude.json 으로 제외됨")
            exp_txt = exp_txt.replace(f"| {m.group(1)} | (대기) |", f"| {m.group(1)} | {score} |", 1)
        if status.endswith("ERROR"):
            issues.append(f"제출 오류: {desc[:80]}")
    for r in rows:
        try:
            best_lb = max(best_lb, float(r.get("publicScore") or 0))
        except ValueError:
            pass
    exp.write_text(exp_txt)

    # 5. 순위와 메달 컷
    lines += ["", "## 순위"]
    tmp = Path(tempfile.mkdtemp())
    sh("kaggle", "competitions", "leaderboard", COMP, "-d", "-p", str(tmp))
    rank = cut_s = cut_g = n = None
    zs = list(tmp.glob("*.zip"))
    if zs:
        with zipfile.ZipFile(zs[0]) as zf:
            name = zf.namelist()[0]
            lb = list(csv.DictReader(io.StringIO(zf.read(name).decode())))
        sc = sorted((float(x["Score"]) for x in lb), reverse=True)
        n = len(sc)
        rank = 1 + sum(s > best_lb for s in sc)
        cut_s = sc[max(0, int(n * 0.05) - 1)]
        cut_g = sc[min(n - 1, 10 + int(n * 0.002) - 1)]
        lines.append(f"- 내 best LB {best_lb:.5f}, 순위 {rank}/{n}, 은 컷 {cut_s:.5f} (차이 {best_lb - cut_s:+.4f}), 금 컷 {cut_g:.5f}, 1위 {sc[0]:.5f}")
        if best_lb < cut_s:
            issues.append(f"은 컷 아래 ({best_lb - cut_s:+.4f})")
    shutil.rmtree(tmp, ignore_errors=True)

    # 6. 디스크
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    lines += ["", "## 자원", f"- 디스크 여유 {free_gb:.1f} GB"]
    if free_gb < 10:
        issues.append(f"디스크 여유 {free_gb:.1f}GB < 10GB")

    lines[1:1] = ["## 문제", *([f"- ⚠ {i}" for i in issues] or ["- 없음"]), ""]
    (OPS / "status.md").write_text("\n".join(lines) + "\n")
    hist = OPS / "history.tsv"
    if not hist.exists():
        hist.write_text("time\tbest_lb\trank\tteams\tsilver_cut\tgold_cut\tsubs_today\tissues\n")
    with hist.open("a") as f:
        f.write(f"{NOW.isoformat(timespec='minutes')}\t{best_lb}\t{rank}\t{n}\t{cut_s}\t{cut_g}\t{n_today}\t{len(issues)}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
