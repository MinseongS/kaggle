"""Operating-loop health check for name-that-llm (mechanical facts only, no judgement).

uv run python src/health.py  ->  prints markdown, writes outputs/daily/health-latest.md, appends HEALTH.md row.
Judgement and fixes are done by the daily Claude run (DAILY.md step 6).
"""
import csv
import datetime as dt
import glob
import io
import os
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

COMP = "name-that-llm"
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "outputs" / "daily"
now = dt.datetime.now(dt.timezone.utc)
today = now.date()


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return "\n".join(l for l in r.stdout.splitlines() if not l.startswith("Warning:"))


out, flags = [], []

# ---- submissions
subs = list(csv.DictReader(io.StringIO(sh(f"kaggle competitions submissions {COMP} -v"))))
for s in subs:
    s["d"] = dt.datetime.fromisoformat(s["date"][:19]).date()
    try:
        s["sc"] = float(s.get("publicScore") or "nan")
    except ValueError:
        s["sc"] = float("nan")
per_day = Counter(s["d"] for s in subs)
last7 = [(today - dt.timedelta(days=i)) for i in range(7)][::-1]
used = [per_day.get(d, 0) for d in last7]
scored = sorted([s for s in subs if s["sc"] == s["sc"]], key=lambda s: s["date"])
best, best_date, run_best = None, None, -1
for s in scored:
    if s["sc"] >= run_best + 0.010 or best is None:  # count only +0.010 jumps as improvement
        best_date = s["d"]
    if s["sc"] > run_best:
        run_best, best = s["sc"], s
pending = [s for s in subs if "PENDING" in s["status"]]
errors = [s for s in subs if "ERROR" in s["status"]]
out.append("## 제출")
out.append(f"- 최근 7일 UTC 일별 제출 수: {', '.join(f'{d:%m-%d}:{u}' for d, u in zip(last7, used))} (한도 5)")
out.append(f"- 오늘(UTC {today}) 사용 {per_day.get(today, 0)}/5, 채점 중 {len(pending)}건, 에러 {len(errors)}건(전체)")
if best:
    stale = (today - best_date).days
    out.append(f"- best LB {best['sc']:.3f} ({best['description'][:60]}), 마지막 +0.010 개선 {best_date} → {stale}일 전")
    if stale >= 3:
        flags.append(f"정체: {stale}일째 +0.010 개선 없음 → 전략 전환 검토")
first = min(per_day) if per_day else today
unused = sum(5 - u for d, u in zip(last7[:-1], used[:-1]) if d >= first)
if unused >= 10:
    flags.append(f"제출 한도 미사용: 지난 6일 {unused}회 남김 → 제출 후보 파이프라인이 부족함")

# ---- leaderboard position
lbdir = ROOT / "outputs" / "lb"
lbdir.mkdir(parents=True, exist_ok=True)
sh(f"kaggle competitions leaderboard {COMP} -d -p {lbdir} -q")
zs = sorted(glob.glob(str(lbdir / "*.zip")), key=os.path.getmtime)
if zs and best:
    with zipfile.ZipFile(zs[-1]) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), "utf-8")))
    sc = sorted((float(r["Score"]) for r in rows if r.get("Score")), reverse=True)
    n = len(sc)
    gold_k, silver_k, bronze_k = 10 + int(0.002 * n), int(0.05 * n), int(0.10 * n)
    rank = 1 + sum(x > best["sc"] for x in sc)
    out.append("## 순위")
    out.append(f"- {rank}위 / {n}팀. 컷: 금 {sc[gold_k-1]:.3f} (#{gold_k}), 은 {sc[silver_k-1]:.3f} (#{silver_k}), 동 {sc[bronze_k-1]:.3f} (#{bronze_k}), 1위 {sc[0]:.3f}")
    for nm, k in (("동", bronze_k), ("은", silver_k), ("금", gold_k)):
        gap = sc[k - 1] - best["sc"]
        out.append(f"  - {nm}까지 {gap:+.3f}" if gap > 0 else f"  - {nm} 컷 안 (여유 {-gap:.3f})")

# ---- public notebooks above our best
kl = list(csv.DictReader(io.StringIO(sh(f"kaggle kernels list --competition {COMP} --sort-by scoreDescending --page-size 20 -v"))))
hi = []
for k in kl:
    m = re.search(r"0\.(\d{3})", k.get("title", ""))
    if m and best and float("0." + m.group(1)) >= best["sc"] + 0.005:
        hi.append(k["ref"])
out.append("## 공개 노트북")
out.append("- 제목상 우리 best보다 높은 것: " + ("; ".join(hi) if hi else "없음"))
qtext = (ROOT / "QUEUE.md").read_text()
todo = [h for h in hi if h.split("/")[1] not in qtext]
if todo:
    flags.append(f"QUEUE에 없는 더 높은 공개 노트북: {', '.join(todo)} → 재현 항목 추가")

# ---- automation runs
logs = sorted(DAILY.glob("run-*.log"))[-6:]
out.append("## 자동 실행")
bad = []
for lg in logs:
    t = lg.read_text(errors="ignore")
    rep = DAILY / lg.name.replace("run-", "report-").replace(".log", ".md")
    prob = []
    if not rep.exists():
        prob.append("리포트 없음")
    if re.search(r"(?i)invalid api key|not logged in|authentication|401|please run /login", t):
        prob.append("인증 실패")
    if len(t) < 200:
        prob.append("로그가 너무 짧음")
    out.append(f"- {lg.stem[4:]}: {'정상' if not prob else ', '.join(prob)}")
    if prob:
        bad.append(lg.stem)
if bad:
    flags.append(f"실행 문제: {', '.join(bad)}")
expected = []
for i in range(3):
    d = dt.date.today() - dt.timedelta(days=i)
    for hm in ("0910", "2110"):
        when = dt.datetime.strptime(f"{d:%Y%m%d}{hm}", "%Y%m%d%H%M")
        if when < dt.datetime.now() - dt.timedelta(hours=1) and when > dt.datetime(2026, 11, 5, 9, 0):
            expected.append(f"{d:%Y%m%d}-{hm}")
# Orca 실행은 run-*.log 없이 report-*.md만 남기고 분이 09:11처럼 밀릴 수 있다 → 날짜+시(HH)로 매칭
have = {p.stem.split("-", 1)[1][:11] for p in list(DAILY.glob("run-*.log")) + list(DAILY.glob("report-*.md"))}
missed = [e for e in expected if e[:11] not in have]
if missed:
    flags.append(f"놓친 실행: {', '.join(missed)} (맥 꺼짐/잠자기?)")
tok = Path.home() / ".claude" / "ntl-oauth-token"
import json
ORCA_ID = "9a4cdb35-e955-4066-89d1-7d2e45c78f3b"
try:
    runs = json.loads(sh(f"orca automations runs --id {ORCA_ID} --json"))["result"]["runs"]
    st = Counter(r.get("status") for r in runs[:6])
    out.append(f"- Orca 자동화 최근 실행 상태: {dict(st) or '아직 없음'}")
    badr = [r for r in runs[:4] if r.get("status") not in ("completed", "succeeded", "running", "dispatched", None)]
    if badr:
        flags.append(f"Orca 실행 이상: {[r.get('status') for r in badr]}")
except Exception as e:  # noqa: BLE001
    flags.append(f"Orca 자동화 조회 실패: {e!r}"[:120])

# ---- queue / experiment throughput
q = (ROOT / "QUEUE.md").read_text()
sec = re.search(r"## 다음 제출.*?\n(.*?)\n## ", q, re.S)
nq = len(re.findall(r"^\d+\.", sec.group(1), re.M)) if sec else 0
ex = (ROOT / "EXPERIMENTS.md").read_text()
ex_days = Counter(m.group(1) for m in re.finditer(r"^\| (\d{4}-\d\d-\d\d) \| e", ex, re.M))
out.append("## 실험 흐름")
out.append(f"- 다음 제출 큐 {nq}개. 최근 로컬 실험 수: " + ", ".join(f"{d}:{ex_days.get(str(d), 0)}" for d in last7[-3:]))
if nq < 2:
    flags.append(f"제출 큐 {nq}개: 내일 쓸 제출 후보 부족 → 로컬 실험에서 후보를 만들 것")
if sum(ex_days.get(str(d), 0) for d in last7[-2:]) == 0:
    flags.append("최근 2일 로컬 실험 없음")

md = f"# 운영 점검 {dt.datetime.now():%Y-%m-%d %H:%M} KST\n\n## 경고\n" + (
    "\n".join(f"- {f}" for f in flags) if flags else "- 없음") + "\n\n" + "\n".join(out) + "\n"
print(md)
(DAILY / "health-latest.md").write_text(md)
hp = ROOT / "HEALTH.md"
if not hp.exists():
    hp.write_text("# 운영 점검 기록\n\n| 시각 | best LB | 오늘 제출 | 큐 | 경고 |\n|---|---|---|---|---|\n")
with hp.open("a") as f:
    f.write(f"| {dt.datetime.now():%m-%d %H:%M} | {best['sc'] if best else '-'} | {per_day.get(today, 0)}/5 | {nq} | {'; '.join(flags) or '-'} |\n")
