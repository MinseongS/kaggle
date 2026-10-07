import random

CONCEPT = ("count the separate objects of each colour (not cells) and draw a bar chart, one bar per colour with "
           "length = object count, sorted from most to fewest; bar orientation fixed per task")


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
    orient = rng.choice(["rows", "cols"])

    def make():
        while True:
            h, w = rng.randint(12, 20), rng.randint(12, 20)
            g = [[0] * w for _ in range(h)]
            k = rng.randint(2, 4)
            cols = rng.sample(range(1, 10), k)
            counts = rng.sample(range(1, 7), k)
            blocked, ok = set(), True
            cellcount = {}
            for col, cnt in zip(cols, counts):
                for _ in range(cnt):
                    bh, bw = rng.randint(1, 3), rng.randint(1, 3)
                    shape = blob(rng, rng.randint(1, bh * bw), bh, bw)
                    sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
                    for _try in range(30):
                        r0, c0 = rng.randint(0, h - sh), rng.randint(0, w - sw)
                        cells = [(r0 + r, c0 + c) for r, c in shape]
                        if any(x in blocked for x in cells):
                            continue
                        for r, c in cells:
                            g[r][c] = col
                            blocked |= {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
                        cellcount[col] = cellcount.get(col, 0) + len(cells)
                        break
                    else:
                        ok = False
            if not ok:
                continue
            order = sorted(zip(counts, cols), reverse=True)
            m = order[0][0]
            out = [[0] * m for _ in range(k)]
            for i, (cnt, col) in enumerate(order):
                for j in range(cnt):
                    out[i][j] = col
            if orient == "cols":
                out = [list(r) for r in zip(*out)][::-1]
            byc = sorted(cols, key=lambda c: -cellcount[c])
            return g, out, byc != [c for _, c in order]

    n = rng.randint(3, 4)
    while True:
        res = [make() for _ in range(n + 1)]
        pairs = [{"input": g, "output": o} for g, o, _ in res]
        if any(m for _, _, m in res[:-1]) and all(p["output"] != pairs[-1]["output"] for p in pairs[:-1]):
            return {"train": pairs[:-1], "test": pairs[-1:]}
