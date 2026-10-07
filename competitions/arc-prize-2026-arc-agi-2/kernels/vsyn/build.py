"""Assemble the Kaggle notebook from src/*.py. Run: uv run python kernels/vsyn/build.py"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--eval", action="store_true",
                help="separate kernel whose commit run solves all 120 eval tasks (local CV); not for submission")
ap.add_argument("--seed-offset", type=int, default=0)
ap.add_argument("--synth-dataset", default="mins00/arc26-synth-val")
ap.add_argument("--eval-hours", type=float, default=6.0,
                help="commit-run budget; 6h for 120 tasks matches the rerun's ~11.8h for 240")
args = ap.parse_args()

HERE = Path(__file__).parent
SLUG = "arc26-synth-eval"
TITLE = SLUG


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def writefile(name):
    return code(f"%%writefile {name}\n" + (HERE / "src" / name).read_text())


cells = [
    {"cell_type": "markdown", "metadata": {}, "source": (
        "# ARC26 synthetic leak-free eval — v8 pipeline (24 decode views, 16 scoring views) on our own generated tasks\n"
        "NVARC 2025 (via the public LB33.89 perfpatch notebook) with: crc32 scoring seeds + PYTHONHASHSEED,\n"
        "per-task exception handling, TTT deadline guard, cheap-first task order, identity fallback for empty outputs.\n"
        "Pass 2 spends leftover time re-solving the lowest vote-margin tasks with new seeds; votes are pooled by score_kgmon.")},
    code("import os, time\n"
         "rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN')\n"
         f"global_end_time = time.time() + (12 * 3600 - 600 if rerun else {args.eval_hours if args.eval else 12} * 3600 - 600)"),
    # v2's first commit runs died with FileNotFoundError on the hard-coded competition path (v1 worked a day
    # earlier), so locate inputs by filename under /kaggle/input and hand them to the scripts via env vars.
    code("import os, glob\n"
         "!ls /kaggle/input /kaggle/input/* | head -50\n"
         "def _find(pattern):\n"
         "    hits = sorted(glob.glob('/kaggle/input/**/' + pattern, recursive=True), key=len)\n"
         "    return hits[0] if hits else None\n"
         "f = _find('arc-agi_test_challenges.json')\n"
         "if f: os.environ['ARC_COMP_DIR'] = os.path.dirname(f)\n"
         "f = _find('qwen3_4b_grids15_sft139/**/config.json')\n"
         "if f: os.environ['ARC_MODEL_DIR'] = os.path.dirname(f)\n"
         "print('ARC_COMP_DIR', os.environ.get('ARC_COMP_DIR'), 'ARC_MODEL_DIR', os.environ.get('ARC_MODEL_DIR'))"),
    code("import glob, os\n"
         "m = glob.glob('/kaggle/input/**/synth_set.marker', recursive=True)\n"
         "assert m, 'synthetic set dataset not attached'\n"
         "os.environ['ARC_COMP_DIR'] = os.path.dirname(m[0])\n"
         "os.environ['ARC_SCORE_AUG_N'] = '2'\n"
         "print('synthetic set:', os.environ['ARC_COMP_DIR'], sorted(os.listdir(os.environ['ARC_COMP_DIR'])))"),
    code("!pip uninstall -y tensorflow"),
    writefile("arc_loader.py"),
    writefile("arc_decoder.py"),
    writefile("arc_solver.py"),
    writefile("starter.py"),
    code(f"import os\nos.environ['ARC_SEED_OFFSET'] = '{args.seed_offset}'"),
    code("import os\nos.environ.setdefault('ARC_EVAL_KEYS', '" + ("all" if args.eval else "debug") + "')\n"
         "!PYTHONHASHSEED=0 UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 "
         "python starter.py --end-time {global_end_time}"),
    writefile("make_submission.py"),
    code("!PYTHONHASHSEED=0 python make_submission.py"),
]

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 4,
}
out = HERE / "notebook"
out.mkdir(exist_ok=True)
(out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"mins00/{SLUG}",
    "title": TITLE,
    "code_file": f"{SLUG}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [args.synth_dataset],
    "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
    "competition_sources": ["arc-prize-2026-arc-agi-2"],
    "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
    "machine_shape": "NvidiaL4",
}, indent=2))
print("built", out)
