"""Frozen LM hidden-state features: mean-pooled over tokens at several layers.

uv run python src/embed.py --model Qwen/Qwen3-1.7B-Base --pool 2500
Writes outputs/emb_<tag>_{train,pool}.npy  (n, n_layers, hidden) fp16, + pool ids.
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from cvlib import OUT, load_pool, load_train

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
ap.add_argument("--layers", default="2,4,6,8,10,14,20,28")
ap.add_argument("--pool", type=int, default=2500)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max_len", type=int, default=384)
a = ap.parse_args()
layers = [int(x) for x in a.layers.split(",")]
tag = a.model.split("/")[-1]

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "right"
model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16).to(dev).eval()
n_layers = model.config.num_hidden_layers
layers = [l for l in layers if l <= n_layers]
print(tag, "layers", layers, "of", n_layers, flush=True)
# only run as deep as needed
model.layers = model.layers[: max(layers)]


@torch.no_grad()
def embed(texts: list[str]) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), len(layers), model.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    for s in range(0, len(texts), a.bs):
        idx = order[s : s + a.bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=a.max_len).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
        m[:, 0] = 0  # drop first token (attention sink)
        for j, l in enumerate(layers):
            h = hs[l].to(torch.float32)
            out[idx, j] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy().astype(np.float16)
        if (s // a.bs) % 200 == 0:
            print(f"{s}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    return out


tr = load_train()
np.save(OUT / f"emb_{tag}_train.npy", embed(tr.text.tolist()))
if a.pool:
    pool = load_pool(a.pool)
    np.save(OUT / f"emb_{tag}_pool{a.pool}.npy", embed(pool.text.tolist()))
    pool[["id", "model"]].to_csv(OUT / f"emb_{tag}_pool{a.pool}_ids.csv", index=False)
print("done")
