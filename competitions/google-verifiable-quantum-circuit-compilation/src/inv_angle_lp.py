"""invsqrt: 비-Clifford Z_POW 회전 각도를 LP 로 재조정해 min_x (tol(x) - |err(x)|) 를 키운다 → rotation eps ↑ → T ↓.
회로 구조/Toffoli/qubit 은 그대로, 각도 값만 바꾼다.

1) 모든 x (1 shot) 시뮬레이션으로 err0(x) 계산 2) 각 회전 i 를 섭동해 b_i(x) (그 회전이 걸리는 비트값) 추출
3) cutting-plane LP: max m s.t. |err0 + B d| <= tol - m 4) 새 kmx 작성.
Usage: uv run python src/inv_angle_lp.py <in.kmb|kmx> <out.kmb> [--workers 12]"""
import argparse, io, math, os, re, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import kickmix as km
import concurrent.futures as cf, multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/verifiers"))
from invsqrt.verify_submission_invsqrt import (ProblemSpecInvSqrt, reference_values, half_turns_to_units,
                                               to_bits, from_bits, z_pow_class, N_P)  # noqa: E402

SPEC = ProblemSpecInvSqrt.for_n_p(N_P)
BATCH = 4096
_G = {}


def load_lines(path):
    c = km.Circuit.from_file(path)
    buf = io.StringIO() if False else None
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"inv_{os.getpid()}.kmx"
    c.to_file(str(tmp), format="kmx")
    return tmp.read_text().splitlines()


def rotation_lines(lines):
    out = []
    for i, ln in enumerate(lines):
        p = ln.split()
        if p and p[0] == "Z_POW":
            v = Fraction(p[2])
            if z_pow_class(v) == "rotation":
                out.append(i)
    return out


def circuit_from_lines(lines):
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"inv_{os.getpid()}_{time.time_ns()}.kmx"
    tmp.write_text("\n".join(lines) + "\n")
    c = km.Circuit.from_file(str(tmp), format="kmx")
    tmp.unlink()
    return c


def _chunk(task):
    lines, start, stop, seed = task
    c = _G.get(id(lines)) or circuit_from_lines(lines)
    sim = km.Simulator(batch_size=BATCH, debug_print=False, counting_operations=False, seed=seed)
    regs = c.register_data; x_reg = km.array(list(regs["x"]))
    out = np.zeros(stop - start, dtype=np.uint64)
    for s in range(start, stop, BATCH):
        xs = (np.arange(s, s + BATCH, dtype=np.uint64) % np.uint64(SPEC.max_x)) + np.uint64(1)
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, to_bits(xs, SPEC.width))
        sim.do(c)
        th = np.array([half_turns_to_units(sim.read_shot_phase(i)) for i in range(BATCH)], dtype=np.uint64)
        n = min(BATCH, stop - s)
        out[s - start:s - start + n] = th[:n]
    return out


def phases(lines, workers=12, seed=1):
    """Tracked phase (uint64 units of 2^-64 turn) for x = 1..max_x, one shot each."""
    total = math.ceil(SPEC.max_x / BATCH) * BATCH
    per = math.ceil(total / workers / BATCH) * BATCH
    tasks = [(lines, s, min(s + per, total), seed + i) for i, s in enumerate(range(0, total, per))]
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        parts = list(ex.map(_chunk, tasks))
    return np.concatenate(parts)[:SPEC.max_x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    lines = load_lines(a.src)
    rot = rotation_lines(lines)
    print("rotations", len(rot), flush=True)
    xs = np.arange(1, SPEC.max_x + 1, dtype=np.uint64)
    target, tol = reference_values(SPEC, xs)
    cache = ROOT / "outputs/research/inv_lp_cache.npz"
    th0 = phases(lines, a.workers)
    err0 = (th0 - target).view(np.int64).astype(np.float64)
    tolf = tol.astype(np.float64)
    margin0 = tolf - np.abs(err0)
    print(f"base: min margin {margin0.min():.4g} at x={int(np.argmin(margin0))+1}, worst ratio {np.max(np.abs(err0)/tolf):.6f} ({time.time()-t0:.0f}s)", flush=True)
    # bit extraction: perturb each rotation by delta half-turns
    delta = Fraction(1, 2**40)  # half-turns -> 2^-41 turns = 2^23 units
    unit = 2**23
    Bm = np.zeros((len(rot), SPEC.max_x), dtype=np.int8)
    for j, li in enumerate(rot):
        p = lines[li].split()
        l2 = list(lines); l2[li] = f"{p[0]} {p[1]} {Fraction(p[2]) + delta}"
        # write as decimal of a float-exact fraction: kickmix reads decimal; use exact repr
        v = Fraction(p[2]) + delta
        l2[li] = f"{p[0]} {p[1]} {float(v)!r}" if False else f"{p[0]} {p[1]} {decimal_str(v)}"
        th = phases(l2, a.workers)
        d = (th - th0).view(np.int64)
        b = np.rint(d / unit)
        bad = np.abs(d - b * unit).max()
        Bm[j] = b.astype(np.int8)
        print(f"  rot {j} line {li} {p[1]} angle {float(Fraction(p[2])):.3e}: bits set {int((b!=0).sum())} values {np.unique(b)[:5]} resid {bad} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(cache, err0=err0, tol=tolf, B=Bm, rot=np.array(rot))
    print("saved", cache)


def decimal_str(v: Fraction) -> str:
    # exact decimal expansion of a dyadic fraction
    n, d = v.numerator, v.denominator
    assert d & (d - 1) == 0
    k = d.bit_length() - 1
    s = str(n * 5**k)
    if k == 0: return s
    s = s.rjust(k + 1, "0")
    return s[:-k] + "." + s[-k:]


if __name__ == "__main__":
    main()
