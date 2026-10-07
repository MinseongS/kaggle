import random

CONCEPT = ("exactly one object contains a single odd-coloured cell; crop that object's bounding box and repaint the "
           "whole object in the odd cell's colour")


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
    bg = 0 if rng.random() < 0.8 else rng.randint(1, 9)
    pal = [c for c in range(10) if c != bg]

    def make():
        while True:
            h, w = rng.randint(10, 18), rng.randint(10, 18)
            g = [[bg] * w for _ in range(h)]
            blocked, objs = set(), []
            ocols = rng.sample(pal, rng.randint(1, 3))
            for _ in range(rng.randint(3, 6)):
                bh, bw = rng.randint(2, 5), rng.randint(2, 5)
                shape = blob(rng, rng.randint(max(3, bh * bw // 3), bh * bw), bh, bw)
                sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
                for _try in range(30):
                    r0, c0 = rng.randint(0, h - sh), rng.randint(0, w - sw)
                    cells = [(r0 + r, c0 + c) for r, c in shape]
                    if any(x in blocked for x in cells):
                        continue
                    col = rng.choice(ocols)
                    for r, c in cells:
                        g[r][c] = col
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                blocked.add((r + dr, c + dc))
                    objs.append((cells, col))
                    break
            if len(objs) < 3:
                continue
            cand = [o for o in objs if len(o[0]) >= 4]
            if not cand:
                continue
            cells, col = rng.choice(cand)
            S = rng.choice([c for c in pal if c not in ocols])
            sr, sc = rng.choice(cells)
            g[sr][sc] = S
            r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
            r1, c1 = max(r for r, _ in cells), max(c for _, c in cells)
            out = [[bg] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
            for r, c in cells:
                out[r - r0][c - c0] = S
            return g, out

    n = rng.randint(3, 4)
    while True:
        pairs = [dict(zip(("input", "output"), make())) for _ in range(n + 1)]
        if all(p["output"] != pairs[-1]["output"] for p in pairs[:-1]):
            return {"train": pairs[:-1], "test": pairs[-1:]}
