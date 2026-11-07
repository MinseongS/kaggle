"""one-body 행(N개) 각도 단어를 ±1 좌표 탐색으로 골라 error_1 을 낮춘다. 두-body 단어·w·W 는 고정.
error_1 = || sum_j lam_{pi(j)} v_j v_j^T - h1' ||_F (공식 정의, pi 는 |overlap| 할당)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def decode_row(word, edges, brot):
    v = np.zeros(len(word) + 1); v[0] = 1.0
    th = np.pi * word / (1 << (brot - 1))
    c, s = np.cos(th), np.sin(th)
    for t, (p, q) in enumerate(edges):
        a, b = v[p], v[q]
        v[p] = c[t] * a - s[t] * b; v[q] = s[t] * a + c[t] * b
    return v / np.linalg.norm(v)


def optimize_onebody(words1, h1p, edges, brot, sweeps=4, log=print):
    words1 = words1.copy()
    N = h1p.shape[0]
    lam, vec = np.linalg.eigh(h1p)
    V = np.array([decode_row(w, edges, brot) for w in words1])
    ov = np.abs(V @ vec); ri, ci = linear_sum_assignment(-ov)
    perm = np.empty(N, int); perm[ri] = ci
    L = lam[perm]
    E = np.einsum("j,jp,jq->pq", L, V, V) - h1p
    cur = np.sum(E * E)
    log(f"  onebody start error_1={np.sqrt(cur):.6f}")
    M = 1 << brot
    for sw in range(sweeps):
        improved = 0
        for j in range(N):
            vj = V[j]
            for t in range(N - 1):
                best = None
                for dlt in (1, -1):
                    w = words1[j].copy(); w[t] = (w[t] + dlt) % M
                    vn = decode_row(w, edges, brot)
                    En = E + L[j] * (np.outer(vn, vn) - np.outer(vj, vj))
                    val = np.sum(En * En)
                    if val < cur - 1e-15 and (best is None or val < best[0]):
                        best = (val, w, vn, En)
                if best:
                    cur, words1[j], V[j], E = best[0], best[1], best[2], best[3]
                    vj = V[j]; improved += 1
        log(f"  sweep {sw}: error_1={np.sqrt(cur):.6f} improved={improved}")
        if improved == 0:
            break
    return words1
