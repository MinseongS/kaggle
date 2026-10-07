import random

CONCEPT = ("some border sides are coloured walls; every object slides straight toward the wall of its own colour "
           "until it touches it; objects whose colour matches no wall stay put")

DIRS = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}


def blob(rng, n, bh, bw):
    cells = {(rng.randrange(bh), rng.randrange(bw))}
    while len(cells) < n:
        r, c = rng.choice(sorted(cells))
        dr, dc = rng.choice(((0, 1), (1, 0), (0, -1), (-1, 0)))
        if 0 <= r + dr < bh and 0 <= c + dc < bw:
            cells.add((r + dr, c + dc))
    mr, mc = min(r for r, _ in cells), min(c for _, c in cells)
    return sorted((r - mr, c - mc) for r, c in cells)


def generate(rng: random.Random) -> dict:
    def make():
        while True:
            h, w = rng.randint(12, 18), rng.randint(12, 18)
            sides = rng.sample("TBLR", rng.randint(2, 4))
            cols = rng.sample(range(1, 10), len(sides) + 1)
            wc = dict(zip(sides, cols))
            stray = cols[-1]
            g = [[0] * w for _ in range(h)]
            out = [[0] * w for _ in range(h)]
            for s, col in wc.items():
                if s == "T": cells = [(0, c) for c in range(1, w - 1)]
                elif s == "B": cells = [(h - 1, c) for c in range(1, w - 1)]
                elif s == "L": cells = [(r, 0) for r in range(1, h - 1)]
                else: cells = [(r, w - 1) for r in range(1, h - 1)]
                for r, c in cells:
                    g[r][c] = out[r][c] = col
            used, ok, moved = set(), True, 0
            nobj = rng.randint(3, 6)
            kinds = [rng.choice(sides) for _ in range(nobj)] + (["X"] if rng.random() < 0.5 else [])
            placed = 0
            for kind in kinds:
                for _try in range(30):
                    bh, bw = rng.randint(1, 3), rng.randint(1, 3)
                    shape = blob(rng, rng.randint(1, bh * bw), bh, bw)
                    sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
                    r0, c0 = rng.randint(2, h - 2 - sh), rng.randint(2, w - 2 - sw)
                    cells = [(r0 + r, c0 + c) for r, c in shape]
                    if kind == "X":
                        sweep, final, col = set(cells), cells, stray
                    else:
                        dr, dc = DIRS[kind]
                        if kind == "T": k = min(r for r, _ in cells) - 1
                        elif kind == "B": k = h - 2 - max(r for r, _ in cells)
                        elif kind == "L": k = min(c for _, c in cells) - 1
                        else: k = w - 2 - max(c for _, c in cells)
                        sweep = {(r + dr * i, c + dc * i) for r, c in cells for i in range(k + 1)}
                        final = [(r + dr * k, c + dc * k) for r, c in cells]
                        col = wc[kind]
                    halo = {(r + a, c + b) for r, c in sweep for a in (-1, 0, 1) for b in (-1, 0, 1)}
                    if halo & used:
                        continue
                    used |= sweep
                    for r, c in cells:
                        g[r][c] = col
                    for r, c in final:
                        out[r][c] = col
                    if kind != "X" and k > 0:
                        moved += 1
                    placed += 1
                    break
            if placed >= 3 and moved >= 2:
                return g, out

    n = rng.randint(3, 4)
    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(n + 1)]
        if all(p["output"] != pairs[-1]["output"] for p in pairs[:-1]):
            return {"train": pairs[:-1], "test": pairs[-1:]}
