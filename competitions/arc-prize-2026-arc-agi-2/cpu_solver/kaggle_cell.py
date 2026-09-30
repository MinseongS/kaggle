"""Emit notebook cell sources that build + launch the CPU solver on Kaggle (used by kernels/v3/build.py).

The C++ sources (~50 KB tar.xz) are embedded as base64 so the notebook needs no extra dataset.
Usage from kernels/<ver>/build.py:
    import sys; sys.path.insert(0, str(COMP_DIR / "cpu_solver")); import kaggle_cell
    cells += [code(kaggle_cell.build_and_launch_cell()), ...]; later: code(kaggle_cell.merge_cell())
"""
import base64
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).parent


def _tarball_b64():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tf:
        for p in ["src", "headers", "Makefile", "LICENSE"]:
            tf.add(HERE / "icecuber" / p, arcname=f"icecuber/{p}",
                   filter=lambda ti: None if ti.name.endswith((".gch", ".pch", ".o")) else ti)
        tf.add(HERE / "run_icecuber.py", arcname="run_icecuber.py")
        tf.add(HERE / "merge_cpu.py", arcname="merge_cpu.py")
    return base64.b64encode(buf.getvalue()).decode()


def build_and_launch_cell(workers=16, nice=15, passes="3:900:0,23:1200:0,33:1200:0", mem_mb=6000,
                          total_mem_mb=64000, end_buffer_s=1800):
    """Runs BEFORE the GPU cell. Builds in /tmp (not /kaggle/working: output commit), launches detached.
    Any failure leaves cpu_proc = None and the GPU pipeline untouched."""
    return f'''import base64, io, os, subprocess, tarfile, time, json
CPU_DIR = "/tmp/cpu_solver"
cpu_proc = None
comp_dir = os.environ.get("ARC_COMP_DIR", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
cpu_challenges = f"{{comp_dir}}/arc-agi_test_challenges.json" if os.getenv("KAGGLE_IS_COMPETITION_RERUN") \\
    else f"{{comp_dir}}/arc-agi_evaluation_challenges.json"
try:
    os.makedirs(CPU_DIR, exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(base64.b64decode("{_tarball_b64()}")), mode="r:xz").extractall(CPU_DIR)
    print(subprocess.run("nproc; free -g | head -2; g++ --version | head -1", shell=True, capture_output=True, text=True).stdout)
    t0 = time.time()
    r = subprocess.run(["make", "-j8"], cwd=f"{{CPU_DIR}}/icecuber", capture_output=True, text=True)
    print("cpu solver build", r.returncode, f"{{time.time()-t0:.0f}}s", r.stderr[-2000:])
    if r.returncode == 0:
        cpu_proc = subprocess.Popen(
            ["python", f"{{CPU_DIR}}/run_icecuber.py", "--challenges", cpu_challenges, "--out", f"{{CPU_DIR}}/cpu.json",
             "--workers", "{workers}", "--nice", "{nice}", "--passes", "{passes}", "--mem-mb", "{mem_mb}",
             "--total-mem-mb", "{total_mem_mb}", "--deadline", str(global_end_time - {end_buffer_s})],
            stdout=open(f"{{CPU_DIR}}/cpu.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
except Exception as e:
    print("cpu solver disabled:", repr(e))
'''


def merge_cell():
    """Runs AFTER make_submission.py wrote submission.json. On eval commits also prints before/after scores
    and keeps cpu.json + the pre-merge submission in /kaggle/working for offline merge-rule tuning."""
    return '''import os, subprocess, signal, time, json, shutil
def _score(sub, sol):
    tot = 0.0
    for k, outs in sol.items():
        att = sub.get(k, [])
        tot += sum(i < len(att) and g in (att[i].get("attempt_1"), att[i].get("attempt_2")) for i, g in enumerate(outs)) / len(outs)
    return tot
try:
    if cpu_proc is not None:
        # merge even if passes are still going: kill, then re-run the driver with no passes to merge what exists
        if cpu_proc.poll() is None:
            os.killpg(cpu_proc.pid, signal.SIGKILL); time.sleep(1)
        subprocess.run(["python", f"{CPU_DIR}/run_icecuber.py", "--challenges", cpu_challenges,
                        "--out", f"{CPU_DIR}/cpu.json", "--passes", ""], check=False)
        print(open(f"{CPU_DIR}/cpu.log").read()[-3000:])
    if cpu_proc is not None and os.path.exists(f"{CPU_DIR}/cpu.json"):
        rerun = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
        if not rerun:
            shutil.copy("submission.json", "/kaggle/working/submission_nvarc.json")
            shutil.copy(f"{CPU_DIR}/cpu.json", "/kaggle/working/cpu.json")
            shutil.copy(f"{CPU_DIR}/cpu.json.stats.jsonl", "/kaggle/working/cpu_stats.jsonl")
        subprocess.run(["python", f"{CPU_DIR}/merge_cpu.py", "--submission", "submission.json",
                        "--cpu", f"{CPU_DIR}/cpu.json", "--challenges", cpu_challenges,
                        "--out", "submission_merged.json"], check=True)
        merged = json.load(open("submission_merged.json"))
        assert set(merged) == set(json.load(open("submission.json")))
        if not rerun:
            sol = json.load(open(cpu_challenges.replace("_challenges", "_solutions")))
            print("*** score NVARC only:", _score(json.load(open("submission.json")), sol), " merged:", _score(merged, sol))
        shutil.move("submission_merged.json", "submission.json")
except Exception as e:
    print("cpu merge skipped:", repr(e))
'''


if __name__ == "__main__":
    s = build_and_launch_cell()
    print(len(s), "chars in build cell")
