import random

CONCEPT = ("delete every cell of the noise colour, then join each pair of same-coloured points lying on a common "
           "row or column with a straight line of that colour")


def generate(rng: random.Random) -> dict:
    N = rng.randint(1, 9)
    others = [c for c in range(1, 10) if c != N]

    def make():
        while True:
            h, w = rng.randint(12, 20), rng.randint(12, 20)
            g = [[0] * w for _ in range(h)]
            out = [[0] * w for _ in range(h)]
            blocked, ends = set(), set()
            cols = rng.sample(others, rng.randint(3, 5))
            ok = 0
            for col in cols:
                for _try in range(40):
                    L = rng.randint(4, 12)
                    if rng.random() < 0.5:
                        if L >= w: continue
                        r, c0 = rng.randrange(h), rng.randint(0, w - 1 - L)
                        cells = [(r, c) for c in range(c0, c0 + L + 1)]
                    else:
                        if L >= h: continue
                        c, r0 = rng.randrange(w), rng.randint(0, h - 1 - L)
                        cells = [(r, c) for r in range(r0, r0 + L + 1)]
                    if any(x in blocked for x in cells):
                        continue
                    for r, c in cells:
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                blocked.add((r + dr, c + dc))
                        out[r][c] = col
                    for r, c in (cells[0], cells[-1]):
                        g[r][c] = col
                        ends.add((r, c))
                    ok += 1
                    break
            if ok < 3:
                continue
            dens = rng.uniform(0.05, 0.13)
            for r in range(h):
                for c in range(w):
                    if (r, c) not in ends and rng.random() < dens:
                        g[r][c] = N
            return g, out

    n = rng.randint(3, 4)
    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(n + 1)]
        if all(p["output"] != pairs[-1]["output"] for p in pairs[:-1]):
            return {"train": pairs[:-1], "test": pairs[-1:]}
