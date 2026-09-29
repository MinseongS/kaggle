"""Assemble the Kaggle FPNet training notebook(s) from src/casmi.

Run: uv run python kernels/train_fpnet/build.py --variant merged|single [--steps 30000 --max-minutes 330]
     kaggle kernels push -p kernels/train_fpnet/<variant>
Inputs: mins00/casmi26-fptrain-v1 (spec.npz / pool_*.npy from `python -m casmi.train_fpnet prep`, i.e. training
spectra and decoy pool with every holdout_v1.json metric key removed).
Output: /kaggle/working/fp_<variant>_h1.pt (+ .hist.json), loadable by fpnet.ModelBank.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["merged", "single"], required=True)
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--max-minutes", type=float, default=330)
ap.add_argument("--tims-frac", type=float, default=0.6)
ap.add_argument("--tag", default="h1")
args = ap.parse_args()

HERE = Path(__file__).parent
SRC = HERE.parents[1] / "src" / "casmi"
SLUG = f"casmi26-train-fpnet-{args.variant}"
DATASET = "mins00/casmi26-fptrain-v1"
MERGE_P = 0.6 if args.variant == "merged" else 0.0


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


train_cmd = (f"--data {{DATA}} --out /kaggle/working/fp_{args.variant}_{args.tag}.pt --merge_p {MERGE_P} "
             f"--steps {args.steps} --max_minutes {args.max_minutes} --tims_frac {args.tims_frac} --workers 3")
cells = [
    {"cell_type": "markdown", "metadata": {}, "source": f"# FPNet training ({args.variant}, leak-free holdout_v1)\n"
     "prvsiyan recipe (BCE + 63 same-window decoys, best-val checkpoint); `src/casmi/train_fpnet.py`."},
    code("!mkdir -p /kaggle/working/casmi && nvidia-smi"),
]
for name in ["__init__.py", "fpnet.py", "train_fpnet.py"]:
    cells.append(code(f"%%writefile /kaggle/working/casmi/{name}\n" + (SRC / name).read_text()))
cells += [
    code("import glob, os\n"
         "DATA = os.path.dirname(glob.glob('/kaggle/input/**/spec.npz', recursive=True)[0])\n"
         "print(DATA, os.listdir(DATA))"),
    code(f"!cd /kaggle/working && python -m casmi.train_fpnet train {train_cmd}"),
    code("!rm -rf /kaggle/working/casmi && ls -la /kaggle/working"),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
out = HERE / args.variant
out.mkdir(exist_ok=True)
(out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"mins00/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": False, "dataset_sources": [DATASET], "kernel_sources": [], "competition_sources": [],
    "model_sources": [], "machine_shape": "NvidiaTeslaT4",
}, indent=2))
print("built", out, len(cells), "cells")
