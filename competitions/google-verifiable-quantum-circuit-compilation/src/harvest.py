"""공개 노트북 출력 + 자체 후보를 모아 과제별 최선 회로로 포트폴리오를 만들고, 개선되면 제출한다.

Usage:
    uv run python src/harvest.py [--top 15] [--no-submit] [--skip-download]

흐름:
  1. 점수 상위 / 최근 공개 노트북의 submission.zip 을 outputs/harvest/kernels/<owner>__<slug>/ 로 받는다.
  2. outputs/candidates/<name>/ (자체 연구 결과, 과제 파일 일부만 있어도 됨) 도 후보에 넣는다.
  3. 과제별 파일 해시로 캐시하며 공식 verifier 로 검증한다 (dqi 는 근사 디코더가 있어 랜덤 salt 로 한 번 더 돌려 나쁜 쪽 비용을 쓴다).
     dqi 최종 후보는 salt 를 ROBUST_SALTS 개까지 늘려 가장 나쁜 비용을 쓴다 (2026-11-07: 공개 salt 에서만 실패 0 인
     회로가 LB 비공개 salt 에서 2샷 실패 → R 1.191→1.201, LB −0.07). LB 로 확인된 불량 회로는 lb_exclude.json 으로 뺀다.
  4. 과제별 최소 비용 회로를 골라 outputs/harvest/best/ 에 조립, 로컬 GM 이 기준선(LB 로 확인된 최고 점수, 채점 대기 중인
     제출의 로컬 GM 중 큰 값)보다 높고 같은 파일 조합을 낸 적이 없으면 제출.

다운로드한 파일은 데이터로만 다룬다 (.kmb/.npz 를 verifier 가 읽을 뿐, 노트북 코드는 실행하지 않음).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import secrets
import shutil
import subprocess
import zipfile
from pathlib import Path

COMP = "google-verifiable-quantum-circuit-compilation"
ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = ROOT / "data" / "verifiers"
H = ROOT / "outputs" / "harvest"
CACHE = H / "verify_cache.json"
STATE = H / "state.json"
LOG = H / "harvest.log"
REF = {"invsqrt": 2076774.233715282, "dqi": 24439463583672.93, "dfthcsos": 3355599588.3155365}
MAX_AUTO_SUBMITS_PER_DAY = 4  # 1회는 수동용으로 남긴다
MIN_GAIN = 1e-4  # GM 이 이만큼 이상 올라야 제출
ROBUST_SALTS = 4  # dqi 최종 후보에 돌리는 랜덤 salt 수 (실패율이 0 이 아닌 회로 걸러내기)
ROBUST_TOP = 4  # salt 를 늘려 볼 dqi 후보 수 상한 (dqi 검증 1회 ~1.5분)
LB_GAP_EXCLUDE = 0.01  # LB 가 로컬 GM 보다 이만큼 낮으면 그 제출의 dqi 를 제외
EXCLUDE = H / "lb_exclude.json"  # {"<task>:<sha>": "이유"} LB 에서 로컬보다 나쁘게 나온 회로


def log(msg: str) -> None:
    line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def sha_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def task_files(sub: Path, task: str) -> list[Path] | None:
    """제출 폴더에서 과제의 파일 목록. 없으면 None."""
    if task in ("invsqrt", "dqi"):
        for ext in ("kmb", "kmx"):
            p = sub / f"{task}.{ext}"
            if p.is_file():
                return [p]
        return None
    d = sub / "dfthcsos"
    if not (d / "submission.npz").is_file():
        return None
    angles = sorted(d.glob("angle_k_*.km[bx]"), key=lambda p: int(p.stem.split("_")[-1]))
    return angles + [d / "submission.npz"] if angles else None


def run_verifier(task: str, files: list[Path], salt: str | None) -> dict:
    if task == "invsqrt":
        args = ["invsqrt/verify_submission_invsqrt.py", str(files[0])]
    elif task == "dqi":
        args = ["dqi/verify_submission_dqi.py", str(files[0])]
    else:
        args = ["dfthcsos/verify.py", str(files[0].parent)]
    if salt:
        args += ["--salt", salt]
    r = subprocess.run(["uv", "run", "--project", str(ROOT), "python", *args],
                       cwd=VERIFIERS, capture_output=True, text=True, timeout=3600)
    try:
        out = json.loads(r.stdout)
    except Exception:
        out = {"status": "FAIL", "error": (r.stdout + r.stderr)[-500:]}
    return out


def verify_task(task: str, files: list[Path], cache: dict) -> dict:
    key = f"{task}:{sha_files(files)}"
    if key in cache:
        return cache[key]
    log(f"verify {task} {files[0].parent.name if task != 'dfthcsos' else files[0].parent.parent.name} ...")
    res = run_verifier(task, files, None)
    entry = {"status": res.get("status"), "cost": res.get("score"), "error": res.get("error"),
             "Q": res.get("num_qubits"), "T": res.get("toffoli_count")}
    if task == "dqi" and entry["status"] == "PASS":
        res2 = run_verifier(task, files, secrets.token_hex(8))
        if res2.get("status") != "PASS":
            entry = {**entry, "status": "FAIL", "error": f"salted run: {res2.get('error')}"}
        else:
            entry["cost"] = max(entry["cost"], res2["score"])
            entry["salted_cost"] = res2["score"]
    cache[key] = entry
    CACHE.write_text(json.dumps(cache, indent=1))
    return entry


def harden(task: str, files: list[Path], cache: dict) -> dict:
    """salt 를 ROBUST_SALTS 개까지 늘려 돌리고 가장 나쁜 비용을 쓴다. 하나라도 FAIL 이면 FAIL."""
    key = f"{task}:{sha_files(files)}"
    entry = cache[key]
    costs = entry.setdefault("salted_costs", [entry["salted_cost"]] if entry.get("salted_cost") else [])
    while entry["status"] == "PASS" and len(costs) < ROBUST_SALTS:
        log(f"harden {task} {files[0].parent.name} salt {len(costs) + 1}/{ROBUST_SALTS} ...")
        res = run_verifier(task, files, secrets.token_hex(8))
        if res.get("status") != "PASS":
            entry["status"], entry["error"] = "FAIL", f"salted run: {res.get('error')}"
            break
        costs.append(res["score"])
        entry["cost"] = max(entry["cost"], res["score"])
    cache[key] = entry
    CACHE.write_text(json.dumps(cache, indent=1))
    return entry


def lb_baseline(state: dict, exclude: dict) -> tuple[float, set[str]]:
    """기준선 = max(LB 로 확인된 최고, 채점 대기 중 제출의 로컬 GM). 실패하면 state 값."""
    r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv"], capture_output=True, text=True)
    import csv, io, re
    rows = list(csv.DictReader(io.StringIO("\n".join(l for l in r.stdout.splitlines() if not l.startswith("Warning")))))
    if not rows:
        return state["best_gm"], set()
    base, pending = 0.0, set()
    for row in rows:
        score = row.get("publicScore") or ""
        try:
            base = max(base, float(score))
        except ValueError:
            m = re.search(r"GM (\d+\.\d+)", row.get("description", ""))
            if m and "PENDING" in row.get("status", "").upper():
                base = max(base, float(m.group(1)))
                pending.add(m.group(1))
            continue
        # LB 가 로컬보다 확실히 낮으면 그 제출의 dqi 를 제외 목록에 (dqi 만 비용이 salt 에 따라 크게 변함)
        m = re.search(r"auto-harvest GM (\d+\.\d+)", row.get("description", ""))
        if m and float(score) < float(m.group(1)) - LB_GAP_EXCLUDE:
            for h in state.get("history", []):
                if f"{h['gm']:.5f}" == m.group(1):
                    for k in h.get("keys", []):
                        if k.startswith("dqi:") and k not in exclude:
                            exclude[k] = f"LB {score} < local {m.group(1)} ({h['sources'].get('dqi')})"
                            EXCLUDE.write_text(json.dumps(exclude, indent=1))
                            log(f"exclude {k[:20]}: {exclude[k]}")
    return base, pending


def download_kernels(top: int) -> None:
    refs: list[str] = []
    for sort in ("scoreDescending", "dateRun"):
        r = subprocess.run(["kaggle", "kernels", "list", "--competition", COMP, "--sort-by", sort,
                            "--page-size", str(top), "--csv"], capture_output=True, text=True)
        for line in r.stdout.splitlines()[1:]:
            ref = line.split(",")[0]
            if "/" in ref and ref not in refs:
                refs.append(ref)
    for ref in refs:
        d = H / "kernels" / ref.replace("/", "__")
        tmp = d.with_suffix(".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        r = subprocess.run(["kaggle", "kernels", "output", ref, "-p", str(tmp), "--file-pattern", r"submission\.zip"],
                           capture_output=True, text=True, timeout=1800)
        z = tmp / "submission.zip"
        if not z.is_file():
            log(f"no submission.zip in output: {ref}")
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        zh = hashlib.sha256(z.read_bytes()).hexdigest()[:16]
        dest = d / zh
        if dest.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():  # 경로 탈출 방지
                if n.startswith("/") or ".." in Path(n).parts:
                    raise RuntimeError(f"bad zip member {n} in {ref}")
            dest.mkdir(parents=True)
            zf.extractall(dest)
        shutil.rmtree(tmp, ignore_errors=True)
        dedupe(dest)
        log(f"new output {ref} -> {dest.relative_to(ROOT)}")


BLOBS = H / "blobs"


def dedupe(root: Path, min_size: int = 1 << 20) -> None:
    """1MB 이상 파일을 내용 해시 저장소의 하드링크로 바꿔 같은 회로를 한 번만 저장한다."""
    BLOBS.mkdir(parents=True, exist_ok=True)
    for f in root.rglob("*"):
        if not f.is_file() or f.is_symlink() or f.stat().st_size < min_size or f.stat().st_nlink > 1:
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        blob = BLOBS / h.hexdigest()
        if blob.exists():
            tmp = f.with_name(f.name + ".lnk")
            os.link(blob, tmp)
            os.replace(tmp, f)
        else:
            os.link(f, blob)


def prune(keep_sources: set[str]) -> None:
    """현재 최선의 출처도 아니고 커널의 최신 스냅샷도 아닌 스냅샷을 지우고, 안 쓰는 blob 을 지운다."""
    for kdir in (H / "kernels").glob("*"):
        if not kdir.is_dir():
            continue
        if kdir.suffix == ".tmp":
            shutil.rmtree(kdir, ignore_errors=True)
            continue
        snaps = sorted((p for p in kdir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        for s in snaps[:-1]:
            if str(s.relative_to(ROOT)) not in keep_sources:
                shutil.rmtree(s, ignore_errors=True)
                log(f"pruned {s.relative_to(ROOT)}")
    for b in BLOBS.glob("*") if BLOBS.exists() else []:
        if b.stat().st_nlink == 1:
            b.unlink()


def candidate_dirs() -> list[Path]:
    dirs = [p for p in (H / "kernels").glob("*/*") if p.is_dir()]
    dirs += [p for p in (ROOT / "outputs" / "candidates").glob("*") if p.is_dir()]
    dirs.append(ROOT / "outputs" / "public_somejoint" / "sub")
    return [d for d in dirs if d.exists()]


def build_zip(choice: dict[str, list[Path]], out: Path) -> Path:
    shutil.rmtree(out, ignore_errors=True)
    (out / "dfthcsos").mkdir(parents=True)
    for task, files in choice.items():
        for f in files:
            dst = out / ("dfthcsos" if task == "dfthcsos" else "") / f.name
            shutil.copy2(f, dst)
    # 대회 규칙: 제출 파일명은 반드시 submission.zip (아니면 CreateSubmission 400)
    z = out.parent / f"{out.name}_zip" / "submission.zip"
    z.parent.mkdir(parents=True, exist_ok=True)
    z.unlink(missing_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(out).as_posix())
    return z


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    H.mkdir(parents=True, exist_ok=True)

    if not a.skip_download:
        download_kernels(a.top)
    for kdir in (H / "kernels").glob("*/*"):
        if kdir.is_dir():
            dedupe(kdir)

    cache = load_json(CACHE, {})
    exclude = load_json(EXCLUDE, {})
    state = load_json(STATE, {"best_gm": 24.56271, "submits": {}})
    baseline, _ = lb_baseline(state, exclude)  # LB 피드백 → 제외 목록 갱신이 후보 선택보다 먼저
    cands: dict[str, list[tuple[float, list[Path], str]]] = {t: [] for t in REF}
    for d in candidate_dirs():
        for task in REF:
            files = task_files(d, task)
            if not files:
                continue
            if f"{task}:{sha_files(files)}" in exclude:
                continue
            e = verify_task(task, files, cache)
            if e["status"] != "PASS" or not e["cost"]:
                continue
            cands[task].append((e["cost"], files, str(d.relative_to(ROOT))))
    # dqi: 비용 순으로 salt 를 늘려 검증, 검증된 최선이 다음 미검증 후보보다 싸면 멈춘다
    hardened: list[tuple[float, list[Path], str]] = []
    pool = sorted(cands["dqi"], key=lambda c: c[0])
    for c in pool[:ROBUST_TOP]:
        if hardened and min(hardened, key=lambda h: h[0])[0] <= c[0]:
            break
        e = harden("dqi", c[1], cache)
        if e["status"] == "PASS":
            hardened.append((e["cost"], c[1], c[2]))
    cands["dqi"] = hardened + pool[ROBUST_TOP:] if hardened else pool[ROBUST_TOP:]
    best: dict[str, tuple[float, list[Path], str]] = {}
    for task in REF:
        if cands[task]:
            best[task] = min(cands[task], key=lambda c: c[0])
    if len(best) < 3:
        log(f"missing tasks: {set(REF) - set(best)}")
        return

    keep = {best[t][2] for t in REF}
    for h in state.get("history", [])[-3:]:  # 최근 제출 출처는 LB 확인/복구용으로 남긴다
        keep |= set(h.get("sources", {}).values())
    prune(keep)
    ratios = {t: REF[t] / best[t][0] for t in REF}
    gm = math.prod(ratios.values()) ** (1 / 3)
    for t in REF:
        log(f"best {t}: ratio {ratios[t]:.4f} from {best[t][2]}")
    log(f"portfolio local GM {gm:.5f}")

    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    n_today = state["submits"].get(today, 0)
    combo = sorted(f"{t}:{sha_files(best[t][1])}" for t in REF)
    if any(sorted(h.get("keys", [])) == combo for h in state.get("history", [])):
        log(f"same file combination already submitted (GM {gm:.5f}); skip")
        return
    if gm <= baseline + MIN_GAIN:
        log(f"no improvement over baseline {baseline:.5f} (LB best / pending)")
        return
    z = build_zip({t: best[t][1] for t in REF}, H / "best")
    if a.no_submit or n_today >= MAX_AUTO_SUBMITS_PER_DAY:
        log(f"would submit {z} (GM {gm:.5f}); skipped (no-submit={a.no_submit}, today={n_today})")
        return
    msg = "auto-harvest GM %.5f | " % gm + " ".join(f"{t}={ratios[t]:.3f}" for t in REF)
    r = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f", str(z), "-m", msg],
                       capture_output=True, text=True)
    ok = "Successfully" in r.stdout
    log(f"submit {'OK' if ok else 'FAILED'}: {msg} {'' if ok else r.stdout[-300:] + r.stderr[-300:]}")
    if ok:
        state["best_gm"] = gm
        state["submits"][today] = n_today + 1
        state.setdefault("history", []).append({"time": dt.datetime.now().isoformat(timespec="seconds"),
                                                "gm": gm, "ratios": ratios,
                                                "sources": {t: best[t][2] for t in REF}, "keys": combo})
        STATE.write_text(json.dumps(state, indent=1))
        with (ROOT / "EXPERIMENTS.md").open("a") as f:
            f.write(f"| {today} | harvest | 자동 포트폴리오 " + ", ".join(f"{t}←{best[t][2]}" for t in REF)
                    + f" | {gm:.5f} | (대기) | 자동 제출 |\n")


if __name__ == "__main__":
    main()
