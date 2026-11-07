"""형상 평가 파이프라인: 학습 npz → 8비트 양자화 → (w,W) LBFGS 재적합 (e2, gap 제약) → one-body 단어 최적화 → 공식 evaluate_words.
loader 회로는 만들지 않는다. 출력: 공식 지표 + 비용모델 T,Q 추정 + 예상 배율.
Usage: uv run python src/chem_shape_eval.py <trained.npz> [--iters 300] [--out final.npz]"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from chem_quant_eval import quantize, evaluate_words, h1_prime_of, ROOT, reconstruct_u_vectors
from chem_onebody import optimize_onebody
from chem_refit import THR, GAP_FLOOR
from dataclasses import replace
torch.set_default_dtype(torch.float64)
sys.path.insert(0, str(ROOT / "data/verifiers/dfthcsos"))
from scoreinfra.costs_prepare import SpinFreeData_V1  # noqa: E402
REF_COST = 3355599588.3155365


def cost_model(R, B, C, brot=8, J=5, per_circuit=None):
    """angle loader: 공개 품질 정보량 모델 per-circuit ≈ 40A/212 + 73 (A=1220 에서 실측 281 에 맞게 보정)."""
    m = SpinFreeData_V1(R=R, B=B, C=C, N=76, chi=10, beta=brot)
    o, i = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
    Th = m.outer_state_prep() + m.inner_state_prep() + m.givens_rotations()
    A = 76 + R * B
    pc = per_circuit or (40 * A / 212 + 73) * (281.3 / (40 * 1220 / 212 + 73))
    Ta = 2 * 15 * pc
    Q = o["persistent"] + i["persistent"] + J * brot + max(o["temporary"], i["temporary"], 77)
    return Th + Ta, Q


def refit(inp, words, edges, brot, iters, log=print):
    N = inp.ref["oei"].shape[0]; R, B, C = inp.w.shape
    v = reconstruct_u_vectors(words[N:] / (1 << (brot - 1)), edges)
    u2 = torch.tensor((v / np.linalg.norm(v, axis=1, keepdims=True)).reshape(R, B, N))
    tei = np.asarray(inp.ref["tei"], dtype=np.float64); S = inp.S
    T = tei.copy(); di = np.arange(N)
    T[:, :, di, di] += 0.5 * S[:, :, None]; T[di, di, :, :] += 0.5 * S[None, :, :]
    T = T.reshape(N * N, N * N); T = torch.tensor(0.5 * (T + T.T)); Tn2 = float((T * T).sum())
    ref = inp.ref
    h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", tei)
    const_h1 = torch.tensor(h1_thc - 0.5 * ref["electrons"] * S + inp.beta_1 * np.eye(N))
    gap_const = inp.energy_upper - ref["ecore"] + ref["electrons"] * inp.beta_1
    w = torch.tensor(np.ascontiguousarray(inp.w), requires_grad=True); W = torch.tensor(np.ascontiguousarray(inp.W), requires_grad=True)

    def metrics():
        X = torch.einsum("rbp,rbq,rbc->rcpq", u2, u2, w)
        Y = X.reshape(-1, N * N); G = Y @ Y.T
        e2 = torch.sqrt(torch.clamp((G * G).sum() - 2 * ((Y @ T) * Y).sum() + Tn2, min=1e-30))
        h1p = const_h1 - torch.einsum("rc,rcpq->pq", W, X); h1p = 0.5 * (h1p + h1p.T)
        lam = torch.linalg.eigvalsh(h1p)
        lam_h1 = lam.abs().sum(); wB = W + w.sum(1)
        Lam = lam_h1 + 0.25 * ((w.abs().sum(1) + wB.abs()) ** 2).sum()
        gap = gap_const + (lam_h1 - lam.sum()) + 0.5 * (W ** 2).sum()
        return e2, gap, Lam, torch.sqrt(gap * (2 * Lam - gap))

    tgt_e, tgt_g = THR - 3e-4, GAP_FLOOR + 3e-4
    opt = torch.optim.LBFGS([w, W], lr=1, max_iter=20, history_size=20, line_search_fn="strong_wolfe")
    for mu in (1e3, 1e4, 1e5, 1e6):
        for it in range(iters // 4 // 20 + 1):
            def closure():
                opt.zero_grad()
                e2, gap, Lam, le = metrics()
                loss = le + mu * (torch.relu(e2 - tgt_e) ** 2 + torch.relu(tgt_g - gap) ** 2)
                loss.backward(); return loss
            opt.step(closure)
        with torch.no_grad():
            e2, gap, Lam, le = metrics()
        log(f"  refit mu={mu:g}: leff={le.item():.4f} Lam={Lam.item():.3f} gap={gap.item():.6f} e2={e2.item():.6f}")
    return replace(inp, w=w.detach().numpy().copy(), W=W.detach().numpy().copy())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("--iters", type=int, default=400); ap.add_argument("--brot", type=int, default=8)
    ap.add_argument("--out"); ap.add_argument("--sweeps", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    inp, words, edges = quantize(a.npz, a.brot)
    R, B, C = inp.w.shape
    m = evaluate_words(inp, words, edges, a.brot)
    print(f"[{R},{B},{C}] quantized raw: e1={m['error_1']:.5f} e2={m['error_2']:.5f} Lam={m['Lambda']:.3f} gap={m['E_gap']:.5f} leff={m['lambda_eff']:.4f}", flush=True)
    inp = refit(inp, words, edges, a.brot, a.iters)
    N = 76
    words[:N] = optimize_onebody(words[:N], h1_prime_of(inp, words, edges, a.brot), edges, a.brot, sweeps=a.sweeps, log=lambda s: None)
    m = evaluate_words(inp, words, edges, a.brot)
    ok = m["error_1"] <= THR and m["error_2"] <= THR and m["E_gap"] > GAP_FLOOR and m["Lambda"] > 49
    T, Q = cost_model(R, B, C, a.brot)
    score = m["lambda_eff"] * T * Q
    print(f"[{R},{B},{C}] final: e1={m['error_1']:.5f} e2={m['error_2']:.5f} Lam={m['Lambda']:.3f} gap={m['E_gap']:.6f} leff={m['lambda_eff']:.4f} feasible={ok}")
    print(f"[{R},{B},{C}] cost model T={T:.0f} Q={Q} -> predicted ratio {REF_COST/score:.3f} (public best 11.043) ({time.time()-t0:.0f}s)")
    if a.out:
        np.savez(a.out, U=inp.params["U"], W=inp.w, H2_I=inp.W + inp.w.sum(1), H1_I=inp.params["H1_I"],
                 Hsym_lower=inp.params["Hsym_lower"], tree_edges=edges, angle_words=words, brot=np.int64(a.brot))
