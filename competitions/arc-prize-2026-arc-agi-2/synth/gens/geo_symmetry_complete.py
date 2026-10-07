import random

CONCEPT = ("a pattern with mirror symmetry (left-right, top-bottom or both; fixed per task) is partly covered by "
           "blobs of a noise colour; restore every covered cell from its mirror image")


def _ok(pairs):
    if any(p["input"] == p["output"] for p in pairs):
        return False
    if len({str(p["input"]) for p in pairs}) != len(pairs):
        return False
    return all(pairs[-1]["output"] != p["output"] for p in pairs[:-1])


def generate(rng: random.Random) -> dict:
    sym = rng.choice(["lr", "tb", "both"])
    noise = rng.randint(1, 9)
    others = [c for c in range(1, 10) if c != noise]

    def orbit(r, c, h, w):
        s = {(r, c)}
        if sym in ("lr", "both"):
            s |= {(r, w - 1 - c) for r, c in list(s)}
        if sym in ("tb", "both"):
            s |= {(h - 1 - r, c) for r, c in list(s)}
        return s

    def make():
        while True:
            h, w = rng.randint(10, 18), rng.randint(10, 18)
            pal = rng.sample(others, rng.randint(3, 4))
            dens = rng.uniform(0.55, 0.85)
            g = [[0] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    if (r, c) == min(orbit(r, c, h, w)):
                        v = rng.choice(pal) if rng.random() < dens else 0
                        for rr, cc in orbit(r, c, h, w):
                            g[rr][cc] = v
            inp = [row[:] for row in g]
            cov = set()
            for _ in range(rng.randint(1, 3)):
                bh, bw = rng.randint(2, 4), rng.randint(2, 5)
                r0, c0 = rng.randint(0, h - bh), rng.randint(0, w - bw)
                for r in range(r0, r0 + bh):
                    for c in range(c0, c0 + bw):
                        cov.add((r, c))
            if any(orbit(r, c, h, w) <= cov for r, c in cov):
                continue
            for r, c in cov:
                inp[r][c] = noise
            if inp != g:
                return inp, g

    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(rng.randint(3, 4) + 1)]
        if _ok(pairs):
            return {"train": pairs[:-1], "test": pairs[-1:]}
