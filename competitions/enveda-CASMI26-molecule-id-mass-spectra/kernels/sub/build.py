"""Assemble the Kaggle submission notebook from src/casmi.

Run: uv run python kernels/sub/build.py [--ranker rank_train|cv|v2] [--cv-tag exp000]
     v2: uv run python kernels/sub/build.py --ranker v2 --fp-dataset mins00/casmi26-fp-models-h1 [--cpu]
         (needs the private dataset mins00/casmi26-ranker-rows-v2 = rows.npz + meta.json from casmi.rows_v2 build)
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ranker", choices=["rank_train", "cv", "v2"], default="rank_train")
ap.add_argument("--rows-dataset", default="mins00/casmi26-ranker-rows-v2", help="v2: Kaggle dataset with rows.npz + meta.json")
ap.add_argument("--v2-model", choices=["blend", "hgb", "lgb"], default="blend")
ap.add_argument("--v2-cols", choices=["all", "base"], default="all")
ap.add_argument("--cv-tag", default="exp000")
ap.add_argument("--fp-dataset", default=None, help="Kaggle dataset with fp_*.pt to use instead of the public FPNet weights")
ap.add_argument("--cpu", action="store_true", help="CPU session (GPU batch sessions are capped at 2)")
args = ap.parse_args()

HERE = Path(__file__).parent
SRC = HERE.parents[1] / "src" / "casmi"
SLUG = "casmi26-sub"
COMP = "enveda-CASMI26-molecule-id-mass-spectra"
EXT_DATASETS = [
    "prvsiyan/casmi26-fp-models-v2",
    "prvsiyan/casmi26-ranker-features",
    "prvsiyan/chebi-lipidmaps-casmi26",
    "prvsiyan/coconut-casmi26-candidates",
]


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


setup = f'''import glob, os, subprocess, sys
from pathlib import Path

# Offline RDKit pinned to the metric's version.
py = f"cp{{sys.version_info.major}}{{sys.version_info.minor}}"
wheel = glob.glob(f"/kaggle/input/**/rdkit-2026.3.3-{{py}}-*.whl", recursive=True)[0]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-index", wheel], check=True)

# Lay the inputs out the way casmi.config expects: data/{{train,test}}.parquet and data/ext/<dataset>.
data = Path("/kaggle/working/data"); (data / "ext").mkdir(parents=True, exist_ok=True)
comp = Path(sorted(p for p in glob.glob("/kaggle/input/**/test.parquet", recursive=True) if "{COMP}" in p)[0]).parent
for f in ["train.parquet", "test.parquet", "sample_submission.csv"]:
    (data / f).unlink(missing_ok=True); (data / f).symlink_to(comp / f)
for name in {[d.split("/")[1] for d in EXT_DATASETS]!r}:
    src = [p for p in glob.glob(f"/kaggle/input/**/{{name}}", recursive=True) if os.path.isdir(p)][0]
    (data / "ext" / name).unlink(missing_ok=True); (data / "ext" / name).symlink_to(src)
os.environ["CASMI_DATA"] = str(data)
os.environ["CASMI_OUTPUTS"] = "/kaggle/working/outputs"
# Caches are ~1GB; keep them out of /kaggle/working so they aren't saved as notebook output.
os.environ["CASMI_CACHE"] = "/tmp/casmi_cache"
# CV feature rows (for --ranker cv) must sit at outputs/cv/<tag>/feats.pkl.
feats = glob.glob("/kaggle/input/**/casmi26-cv-feats/**/feats.pkl", recursive=True)
if feats:
    dst = Path("/kaggle/working/outputs/cv") / {args.cv_tag!r} / "feats.pkl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True); dst.symlink_to(feats[0])
    print("cv feats:", feats[0])
sys.path.insert(0, "/kaggle/working")
print(comp, sorted(os.listdir(data / "ext")))
FP_DIR = None
if {args.fp_dataset!r} != "None":
    FP_DIR = [p for p in glob.glob("/kaggle/input/**/{(args.fp_dataset or "x/x").split("/")[1]}", recursive=True) if os.path.isdir(p)][0]
    print("fp weights:", FP_DIR, sorted(os.listdir(FP_DIR)))
ROWS_DIR = None
if {args.ranker!r} == "v2":
    ROWS_DIR = str(Path(glob.glob("/kaggle/input/**/{args.rows_dataset.split("/")[1]}/**/rows.npz", recursive=True)[0]).parent)
    print("v2 rows:", ROWS_DIR, sorted(os.listdir(ROWS_DIR)))
'''

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": "# CASMI26 submission\nPorted prvsiyan 4-channel + HistGBM pipeline (`src/casmi` in the repo)."},
    code("!mkdir -p /kaggle/working/casmi"),
]
for f in sorted(SRC.glob("*.py")):
    cells.append(code(f"%%writefile /kaggle/working/casmi/{f.name}\n" + f.read_text()))
cells += [
    code(setup),
    code("import time; t0 = time.time()\n"
         "from casmi.cli import main\n"
         "main(['build'])\n"
         "print(f'build {time.time()-t0:.0f}s')"),
    code(f"RANKER, CV_TAG = {args.ranker!r}, {args.cv_tag!r}\n"
         "t0 = time.time()\n"
         "main(['predict', '--ranker', RANKER, '--cv-tag', CV_TAG, '--device', 'cuda' if __import__('torch').cuda.is_available() else 'cpu',\n"
         "      '--out', '/kaggle/working/submission.csv'] + (['--fp-dir', FP_DIR] if FP_DIR else [])\n"
         f"     + (['--ranker-rows', ROWS_DIR, '--v2-model', {args.v2_model!r}, '--v2-cols', {args.v2_cols!r}] if ROWS_DIR else []))\n"
         "print(f'predict {time.time()-t0:.0f}s')"),
    code("import polars as pl\n"
         "from rdkit import Chem\n"
         "sub = pl.read_csv('/kaggle/working/submission.csv')\n"
         "bad = [s for row in sub['smiles'] for s in row.split(';') if Chem.MolFromSmiles(s) is None]\n"
         "assert not bad, bad[:5]\n"
         "print(sub.shape, 'all SMILES parse')"),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
out = HERE / "notebook"; out.mkdir(exist_ok=True)
(out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": f"mins00/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": not args.cpu, "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": EXT_DATASETS + ["metric/rdkit-2026-3-3-wheel"] + (["mins00/casmi26-cv-feats"] if args.ranker == "cv" else [])
    + ([args.rows_dataset] if args.ranker == "v2" else [])
    + ([args.fp_dataset] if args.fp_dataset else []),
    "kernel_sources": [], "competition_sources": [COMP], "model_sources": [], **({} if args.cpu else {"machine_shape": "NvidiaTeslaT4"}),
}, indent=2))
print("built", out, len(cells), "cells")
