"""학습된 factorization npz → brot 비트 각도 단어 (공개 best 와 같은 tree) → 공식 evaluate_words 로 지표.
Usage: uv run python src/chem_quant_eval.py <trained.npz> [--brot 8] [--out words.npz]"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/verifiers"))
sys.path.insert(0, str(ROOT / "data/starter_kit/starter_kit"))
from dfthcsos.verify import load_inputs, evaluate_words  # noqa: E402
from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors, reconstruct_x  # noqa: E402
from starter.task2_angles.trees import tree_angles  # noqa: E402

PUB = ROOT / "outputs/harvest/kernels/manwithacat__vqcc-hedge-wizardry-notebook/146803e4b81ba936/dfthcsos/submission.npz"


def to_words(t, brot):
    return np.floor(np.mod(t, 2.0) * (1 << (brot - 1)) + 0.5).astype(np.int64) % (1 << brot)


def quantize(npz, brot=8, edges=None):
    inp = load_inputs(str(npz))
    if edges is None:
        edges = np.load(PUB)["tree_edges"]
    U = np.asarray(inp.params["U"], dtype=np.float64)
    R, B, N = U.shape
    w2 = to_words(tree_angles(U.reshape(-1, N), edges), brot)
    v2 = reconstruct_u_vectors(w2 / (1 << (brot - 1)), edges)
    v2 = (v2 / np.linalg.norm(v2, axis=1, keepdims=True)).reshape(R, B, N)
    X = reconstruct_x(v2, inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    ref = inp.ref
    h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", ref["tei"])
    h1p = h1_thc - 0.5 * ref["electrons"] * inp.S + inp.beta_1 * np.eye(N) - np.einsum("rc,rcpq->pq", inp.W, X)
    _, vec = np.linalg.eigh(h1p)
    w1 = to_words(tree_angles(vec.T, edges), brot)
    words = np.vstack([w1, w2])
    return inp, words, edges


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("--brot", type=int, default=8); ap.add_argument("--out")
    a = ap.parse_args()
    inp, words, edges = quantize(a.npz, a.brot)
    m = evaluate_words(inp, words, edges, a.brot)
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()})
    if a.out:
        d = np.load(a.npz)
        arrs = {k: d[k] for k in ("U", "W", "H2_I", "H1_I", "Hsym_lower")}
        np.savez(a.out, **arrs, tree_edges=edges, angle_words=words, brot=np.int64(a.brot))
        print("wrote", a.out)


def h1_prime_of(inp, words, edges, brot):
    N = inp.ref["oei"].shape[0]
    R, B, _ = inp.w.shape
    v = reconstruct_u_vectors(words[N:] / (1 << (brot - 1)), edges)
    v = (v / np.linalg.norm(v, axis=1, keepdims=True)).reshape(R, B, N)
    X = reconstruct_x(v, inp.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    ref = inp.ref
    h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", ref["tei"])
    return h1_thc - 0.5 * ref["electrons"] * inp.S + inp.beta_1 * np.eye(N) - np.einsum("rc,rcpq->pq", inp.W, X)
