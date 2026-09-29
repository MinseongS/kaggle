"""Assemble the Kaggle submission notebook from src/casmi. Run: uv run python kernels/sub/build.py"""
import json
from pathlib import Path

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
sys.path.insert(0, "/kaggle/working")
print(comp, sorted(os.listdir(data / "ext")))
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
    code("t0 = time.time()\n"
         "main(['predict', '--ranker', 'rank_train', '--device', 'cuda' if __import__('torch').cuda.is_available() else 'cpu',\n"
         "      '--out', '/kaggle/working/submission.csv'])\n"
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
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": False, "dataset_sources": EXT_DATASETS + ["metric/rdkit-2026-3-3-wheel"],
    "kernel_sources": [], "competition_sources": [COMP], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
}, indent=2))
print("built", out, len(cells), "cells")
