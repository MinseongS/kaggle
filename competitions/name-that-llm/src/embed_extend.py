"""Extend a saved pool embedding to a larger per-label sample, embedding only the new rows.

uv run python src/embed_extend.py --old 2500 --new 6000 --layers 2,4,6,20
Same seed as load_pool, so the top-`old` sample per label is a subset of top-`new`.
Output keeps the old file's 8-slot layout (2,4,6,8,10,14,20,28); slots not in --layers are zero for new rows,
so use m_emb.py --layers with the slot indices of --layers only (2,4,6,20 -> 0,1,2,6).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from cvlib import OUT, load_pool

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
ap.add_argument("--old", type=int, default=2500)
ap.add_argument("--new", type=int, default=6000)
ap.add_argument("--layers", default="2,4,6,20")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max_len", type=int, default=384)
a = ap.parse_args()
SLOTS = [2, 4, 6, 8, 10, 14, 20, 28]
layers = [int(x) for x in a.layers.split(",")]
tag = a.model.split("/")[-1]
old = np.load(OUT / f"emb_{tag}_pool{a.old}.npy")
old_ids = pd.read_csv(OUT / f"emb_{tag}_pool{a.old}_ids.csv")
assert old.shape[1] == len(SLOTS)
pool = load_pool(a.new)
pos = {i: k for k, i in enumerate(old_ids.id.astype(str))}
pid = pool.id.astype(str)
have = pid.map(pos)
assert have.notna().sum() == len(old_ids), (have.notna().sum(), len(old_ids))
new_idx = np.where(have.isna())[0]
print("reuse", int(have.notna().sum()), "embed", len(new_idx), flush=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval()
model.layers = model.layers[: max(layers) + 1]  # keep one extra so hidden_states[max] is pre-norm (as in embed.py)
texts = pool.text.values[new_idx].tolist()
out = np.zeros((len(pool), len(SLOTS), model.config.hidden_size), np.float16)
out[have.notna().values] = old[have.dropna().astype(int).values]
order = np.argsort([len(t) for t in texts]); t0 = time.time()
with torch.no_grad():
    for s in range(0, len(texts), a.bs):
        idx = order[s:s + a.bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).to(torch.float32); m[:, 0] = 0
        for l in layers:
            h = hs[l].to(torch.float32)
            out[new_idx[idx], SLOTS.index(l)] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy().astype(np.float16)
        if (s // a.bs) % 200 == 0:
            print(f"{s}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
np.save(OUT / f"emb_{tag}_pool{a.new}.npy", out)
pool[["id", "model"]].to_csv(OUT / f"emb_{tag}_pool{a.new}_ids.csv", index=False)
print("done", time.time() - t0)
